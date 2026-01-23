import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# 1. PatchEmbedding: 把 (B,27,128,128) -> [B, 64, 512]
#    (因 patch_size=16, 128/16=8 => 8*8=64 个patch)
# --------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_chans=27, 
                 embed_dim=512, 
                 patch_size=16, 
                 img_size=128):
        super().__init__()
        self.patch_size = patch_size
        # num_patches = 8*8=64
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        patch_dim = in_chans * patch_size * patch_size
        
        # 线性映射: (patch_dim -> embed_dim)
        self.proj = nn.Linear(patch_dim, embed_dim)
        
        # 可学习的位置编码: (1,64,512)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: [B, 27, 128,128]
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 1) 拆成不重叠patch => [B, (H//p)*(W//p), C*p*p]
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, 8, 8, 16,16)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B,8,8,C,16,16)
        x = x.view(B, -1, C * p * p)  # => [B,64, 27*16*16] = [B,64,6912]

        # 2) 线性映射到 embed_dim=512
        x = self.proj(x)  # => [B,64,512]

        # 3) 加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]  # => [B,64,512]
        return x


# --------------------------------------------------------------------
# 2. TransformerBlock: 原图中的标准 ViT Encoder Block
# --------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        # x: [B, N=64, D=512]
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)  # => [B,64,512]
        x = x + attn_out

        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


# --------------------------------------------------------------------
# 3. ViTEncoder: patch embedding + 多层 TransformerBlock (depth=4)
# --------------------------------------------------------------------
class ViTEncoder(nn.Module):
    def __init__(self, 
                 in_chans=27,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 patch_size=16,
                 img_size=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_chans, embed_dim, patch_size, img_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])

    def forward(self, x):
        # => [B,64,512]
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x  # [B,64,512]


# --------------------------------------------------------------------
# 4. U-Net Decoder: 原图 4次上采样 (8->16->32->64->128),
#    但对应我们这里: 先 [B,64,512] => [B,512,8,8], 然后 up1->16, up2->32, up3->64, up4->128
# --------------------------------------------------------------------
class UpBlock(nn.Module):
    """转置卷积 + (Conv+BN+ReLU)*2，与原图类似。"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class UNetDecoder(nn.Module):
    """
    将 [B,64,512] => reshape -> [B,512,8,8],
    再经过4次 upsampling => 最终 [B,64,128,128].
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # sqrt(64)=8 => reshape后 (B,512,8,8)
        self.n_patches_side = 8  
        self.up1 = UpBlock(512, 256)  # 8->16
        self.up2 = UpBlock(256, 128)  # 16->32
        self.up3 = UpBlock(128, 64)   # 32->64
        self.up4 = UpBlock(64, 64)    # 64->128

    def forward(self, x):
        # x: [B,64,512]
        B, N, D = x.shape
        h = w = self.n_patches_side  # 8
        x = x.transpose(1, 2).contiguous()  # => [B,512,64]
        x = x.view(B, D, h, w)              # => [B,512,8,8]

        x = self.up1(x)  # => [B,256,16,16]
        x = self.up2(x)  # => [B,128,32,32]
        x = self.up3(x)  # => [B,64,64,64]
        x = self.up4(x)  # => [B,64,128,128]
        return x


# --------------------------------------------------------------------
# 5. 整体结构：ViT + U-Net，**最后变成二分类** (global pool + linear -> [B,1])
# --------------------------------------------------------------------
class ViTUNetBinaryClassifier(nn.Module):
    """
    输入: (B,27,128,128)
    1) ViT Encoder => [B,64,512]
    2) U-Net Decoder => [B,64,128,128]
    3) Global Average Pool => [B,64]
    4) FC => [B,1] (二分类的 logit)
    """
    def __init__(self,
                 in_chans=27,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 patch_size=16,
                 img_size=128):
        super().__init__()
        # ViT 编码部分
        self.encoder = ViTEncoder(in_chans, embed_dim, depth, 
                                  num_heads, patch_size, img_size)
        # U-Net 解码部分
        self.decoder = UNetDecoder(embed_dim)

        # 最后改为二分类: 全局平均池化 + Linear(64->1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # => [B,64,1,1]
            nn.Flatten(),                 # => [B,64]
            nn.Linear(64, 1)             # => [B,1]
        )

    def forward(self, x):
        # 1) ViT 编码 => [B,64,512]
        x_enc = self.encoder(x)
        # 2) U-Net 解码 => [B,64,128,128]
        x_dec = self.decoder(x_enc)
        # 3) 全局池化 + FC => [B,1] (logit)
        out = self.cls_head(x_dec)
        return out


# --------------------------------------------------------------------
# 6. 测试
# --------------------------------------------------------------------
# if __name__ == "__main__":
#     model = ViTUNetBinaryClassifier(
#         in_chans=27,      # 27 通道
#         embed_dim=512,    # 与原图相同
#         depth=4,          # 4 层 TransformerBlock
#         num_heads=8,
#         patch_size=16,    # 原图使用16
#         img_size=128      # 输入 128x128
#     )

#     x = torch.randn(2, 27, 128, 128)  # batch_size=2
#     logits = model(x)                 # => [2,1]
    
#     print(model)
#     print("Input shape :", x.shape)
#     print("Output shape:", logits.shape)
#     print("Logits      :", logits)
