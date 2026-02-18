import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 1. 설정값 관리 (DataClass 사용)
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024  # 최대 문맥 길이 (Time step)
    vocab_size: int = 50304 # GPT-2 기본 단어장 크기 (50257 -> 64배수로 올림)
    n_layer: int = 12       # 블록(층) 개수
    n_head: int = 12        # 어텐션 헤드 개수
    n_embd: int = 768       # 임베딩 차원 (hidden size)
    dropout: float = 0.0    # 드롭아웃 비율 (pretrain은 0.0 추천)
    bias: bool = True       # True: 편향(bias) 사용 (GPT-2 스타일)

# -----------------------------------------------------------------------------
# 2. 핵심 부품: Causal Self Attention (V100 가속 적용)
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 키, 쿼리, 밸류를 한 번에 계산 (속도 최적화)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 출력 프로젝션
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 정규화
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size() # [Batch, Time, Channels]

        # 1. Q, K, V 분리
        # (B, T, 3*C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 2. Flash Attention (V100 가속 핵심!)
        # 파이토치 2.0 내장 함수 사용 -> 메모리 절약 & 속도 3배
        # is_causal=True 덕분에 마스크(Mask)를 따로 안 만들어도 미래를 못 보게 처리함
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                           dropout_p=self.dropout if self.training else 0, 
                                           is_causal=True)

        # 3. 다시 합치기 (Re-assemble)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 4. 출력 프로젝션
        y = self.resid_dropout(self.c_proj(y))
        return y

# -----------------------------------------------------------------------------
# 3. 핵심 부품: MLP (Feed Forward)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 논문대로 4배 확장 (예: 768 -> 3072 -> 768)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU() # ReLU보다 좋은 GELU 사용
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# -----------------------------------------------------------------------------
# 4. 블록 조립 (Block)
# -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LayerNorm 구조 (x + Attention(Norm(x))) -> 학습 안정성 UP
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# 5. 전체 모델 (GPT)
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 부품들을 딕셔너리로 묶음
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # 단어 임베딩
            wpe = nn.Embedding(config.block_size, config.n_embd), # 위치 임베딩
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 블록 쌓기
            ln_f = nn.LayerNorm(config.n_embd), # 최종 정규화
        ))
        
        # 언어 모델 헤드 (LM Head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (가중치 공유): 입력 임베딩과 출력 헤드의 가중치를 공유함
        # 파라미터 수를 줄이고 성능을 높임 (GPT 표준)
        self.transformer.wte.weight = self.lm_head.weight

        # 가중치 초기화 (Initialization)
        self.apply(self._init_weights)
        
        # 파라미터 수 출력
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        # 모델 파라미터 개수 세기
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # GPT-2 스타일 초기화
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # 1. 위치 정보 생성 (0, 1, 2, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 2. 임베딩 (단어 + 위치)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 3. 블록 통과
        for block in self.transformer.h:
            x = block(x)
            
        # 4. 최종 정규화
        x = self.transformer.ln_f(x)

        # 5. Loss 계산 (학습 시)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 추론 시: 마지막 토큰에 대해서만 계산 (속도 최적화)
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss

    # Weight Decay 설정을 위한 Optimizer 구성 함수 (중요!)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 모든 파라미터를 가져옴
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # 2차원 이상(행렬)은 Decay 적용, 1차원 이하(Bias, Norm)는 Decay 미적용
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # AdamW 생성 (fused 옵션으로 V100 가속)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # 텍스트 생성 함수 (Inference)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 문맥 길이 자르기
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 모델 예측
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # 마지막 토큰만 사용
            
            # Top-K Sampling (선택)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 확률 분포로 변환 후 샘플링
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 결과 붙이기
            idx = torch.cat((idx, idx_next), dim=1)
        return idx