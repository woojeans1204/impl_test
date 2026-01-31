import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """
    DDPM 표준 Time Embedding (Sinusoidal + MLP)
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
        return embeddings

class ResidualBlock(nn.Module):
    """
    DDPM 논문의 "Wide ResNet" 스타일 블록
    Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU
    Time Embedding은 두 번째 Conv 전에 더해짐
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1, num_groups=32):
        super().__init__()
        
        # 첫 번째 블록
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Time Embedding Projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # 두 번째 블록
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Shortcut (채널 수가 다르면 맞춰줌)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        # 1. First Conv
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        # 2. Add Time Embedding
        # t_emb: [Batch, TimeDim] -> [Batch, OutCh] -> [Batch, OutCh, 1, 1]
        h += self.time_proj(t_emb)[:, :, None, None]
        
        # 3. Second Conv
        h = self.act2(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    DDPM 표준 Self-Attention (QKV 방식)
    """
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        # DDPM은 보통 Head당 32채널 혹은 64채널을 사용하나, 여기선 간단히 4헤드 고정
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, -1).transpose(1, 2) # [B, H*W, C]
        
        # Self Attention
        h, _ = self.attn(h, h, h)
        
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h

class Downsample(nn.Module):
    """ Strided Convolution으로 다운샘플링 """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """ Nearest Neighbor Interpolation + Convolution """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DDPMUNet(nn.Module):
    """
    인터페이스는 그대로 유지하되, 내부는 Original DDPM 아키텍처 구현
    CIFAR-10 표준 설정:
    - Base Dim: 128
    - Channel Multipliers: 1, 2, 2, 2
    - Attention Resolutions: 16, 8
    """
    def __init__(self, dim=128, in_out_ch=3, 
                 channel_mults=(1, 2, 2, 2), # 32->16->8->4
                 attn_resolutions=(16, 8)):  # Attention을 적용할 해상도
        super().__init__()
        
        self.dim = dim
        self.in_out_ch = in_out_ch
        
        # 1. Time Embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 2. Initial Conv
        self.inc = nn.Conv2d(in_out_ch, dim, 3, padding=1)
        
        # 3. Down Stages
        self.downs = nn.ModuleList([])
        num_resolutions = len(channel_mults)
        curr_ch = dim
        curr_res = 32 # CIFAR-10 기준 시작 해상도
        
        # Skip connection 채널 저장을 위한 리스트
        self.skip_chs = [dim] 

        for i, mult in enumerate(channel_mults):
            out_ch = dim * mult
            for _ in range(2): # 각 스테이지마다 ResBlock 2개씩
                self.downs.append(ResidualBlock(curr_ch, out_ch, time_dim))
                curr_ch = out_ch
                
                # 특정 해상도에서 Attention 적용
                if curr_res in attn_resolutions:
                    self.downs.append(AttentionBlock(curr_ch))
                
                self.skip_chs.append(curr_ch) # Skip Connection 저장

            # 마지막 스테이지가 아니면 다운샘플링
            if i != num_resolutions - 1:
                self.downs.append(Downsample(curr_ch))
                self.skip_chs.append(curr_ch)
                curr_res //= 2

        # 4. Mid Stage (Bottleneck)
        self.mid = nn.ModuleList([
            ResidualBlock(curr_ch, curr_ch, time_dim),
            AttentionBlock(curr_ch),
            ResidualBlock(curr_ch, curr_ch, time_dim)
        ])

        # 5. Up Stages
        self.ups = nn.ModuleList([])
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = dim * mult
            for _ in range(2 + 1): # ResBlock 2개 + (마지막엔 Upsample용 1개 더 처리 가능하도록)
                # Skip connection pop
                skip_ch = self.skip_chs.pop()
                self.ups.append(ResidualBlock(curr_ch + skip_ch, out_ch, time_dim))
                curr_ch = out_ch
                
                if curr_res in attn_resolutions:
                    self.ups.append(AttentionBlock(curr_ch))
            
            # 마지막 스테이지가 아니면 업샘플링
            if i != 0:
                self.ups.append(Upsample(curr_ch))
                curr_res *= 2

        # 6. Final Conv
        self.final_norm = nn.GroupNorm(32, curr_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(curr_ch, in_out_ch, 3, padding=1)

    def forward(self, x, t):
        # Time Embedding
        t = self.time_mlp(t)
        
        # Initial
        x = self.inc(x)
        
        # Skip Connections 저장소
        skips = [x]
        
        # [Down Path]
        for layer in self.downs:
            if isinstance(layer, Downsample):
                x = layer(x)
                skips.append(x)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, t)
                skips.append(x)
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
                # Attention 후에는 skip 저장 안 함 (구조에 따라 다름, 여기선 ResBlock 출력을 주로 사용)
                # 하지만 정확한 구현을 위해 여기서는 덮어쓰기 형태로 진행
            
        # [Mid Path]
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
                
        # [Up Path]
        for layer in self.ups:
            if isinstance(layer, Upsample):
                x = layer(x)
            elif isinstance(layer, ResidualBlock):
                # Skip Connection 연결
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)
                x = layer(x, t)
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
                
        # Final
        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)