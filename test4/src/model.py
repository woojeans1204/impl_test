import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """
    DDPM UNet에서 쓰던 것과 동일한 Time Embedding (Sinusoidal + MLP)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 홀수 차원일 경우 패딩 처리 (안정성)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings

class TextDiffusionTransformer(nn.Module):
    """
    1D Transformer for Diffusion
    입력: (Batch, Seq_Len, Dim) -> 출력: (Batch, Seq_Len, Dim)
    """
    def __init__(self, vocab_size, seq_len, dim=128, depth=6, heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # 1. Token Embedding (Discrete -> Continuous)
        # 이 레이어는 Diffusion 과정 자체에는 참여하지 않지만, 
        # 정수(Index)를 벡터로 바꿀 때 Trainer에서 호출됩니다.
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # 2. Positional Embedding (Learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))
        
        # 3. Time Embedding MLP
        # UNet처럼 시간 정보를 벡터로 변환
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # 4. Transformer Backbone (The "UNet" replacement)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=dim * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True, # (Batch, Seq, Dim) 형태 유지
            norm_first=True   # Pre-Norm (학습 안정성)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 5. Output Projection
        # 들어온 차원 그대로 나갑니다 (Predicted Noise)
        self.final_norm = nn.LayerNorm(dim)
        self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, t):
        """
        x: (Batch, Seq_Len, Dim) <- 노이즈가 섞인 임베딩 벡터
        t: (Batch,) <- 타임스텝
        """
        # 1. Time Embedding 처리
        # (Batch,) -> (Batch, Dim)
        t_emb = self.time_mlp(t)
        
        # 2. 입력에 Positional Embedding과 Time Embedding 더하기
        # t_emb를 (Batch, 1, Dim)으로 바꿔서 시퀀스 전체에 브로드캐스팅
        x = x + self.pos_emb + t_emb.unsqueeze(1)
        
        # 3. Transformer 통과
        x = self.transformer(x)
        
        # 4. 최종 출력
        x = self.final_norm(x)
        x = self.final_linear(x)
        
        return x