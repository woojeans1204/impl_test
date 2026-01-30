import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.relu(self.bnorm1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t_emb))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.relu(self.bnorm2(self.conv2(h)))
        return h

class SimpleUNet(nn.Module):
    def __init__(self, dim=64, in_out_ch=3): # 기본값을 3채널, dim=64로 상향
        super().__init__()
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.ReLU()
        )
        
        # Encoder
        self.down1 = Block(in_out_ch, dim, time_dim)
        self.down2 = Block(dim, dim * 2, time_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = Block(dim * 2, dim * 2, time_dim)

        # Decoder
        self.up1_conv = nn.ConvTranspose2d(dim * 2, dim * 2, 2, stride=2)
        self.up1_block = Block(dim * 4, dim, time_dim) 
        
        self.up2_conv = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up2_block = Block(dim * 2, dim, time_dim)
        
        self.final_conv = nn.Conv2d(dim, in_out_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        h1 = self.down1(x, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h_mid = self.mid(self.pool(h2), t_emb)
        
        h_up1 = self.up1_conv(h_mid)
        h_up1 = self.up1_block(torch.cat((h_up1, h2), dim=1), t_emb)
        
        h_up2 = self.up2_conv(h_up1)
        h_up2 = self.up2_block(torch.cat((h_up2, h1), dim=1), t_emb)
        
        return self.final_conv(h_up2)
