import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Translated comment
# Translated comment
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
        
        # Translated comment
        self.proj = nn.Linear(patch_dim, embed_dim)
        
        # Translated comment
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: [B, 27, 128,128]
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Translated comment
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, 8, 8, 16,16)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B,8,8,C,16,16)
        x = x.view(B, -1, C * p * p)  # => [B,64, 27*16*16] = [B,64,6912]

        # Translated comment
        x = self.proj(x)  # => [B,64,512]

        # Translated comment
        x = x + self.pos_embed[:, :x.size(1), :]  # => [B,64,512]
        return x


# --------------------------------------------------------------------
# Translated comment
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
# Translated comment
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
# Translated comment
# Translated comment
# --------------------------------------------------------------------
class UpBlock(nn.Module):
    """Translated to English."""
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
 [B,64,512] => reshape -> [B,512,8,8],
 4 upsampling => [B,64,128,128].
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # Translated comment
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
# Translated comment
# --------------------------------------------------------------------
class ViTUNetBinaryClassifier(nn.Module):
    """
    input: (B,27,128,128)
    1) ViT Encoder => [B,64,512]
    2) U-Net Decoder => [B,64,128,128]
    3) Global Average Pool => [B,64]
 4) FC => [B,1] (classification logit)
    """
    def __init__(self,
                 in_chans=27,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 patch_size=16,
                 img_size=128):
        super().__init__()
        # Translated comment
        self.encoder = ViTEncoder(in_chans, embed_dim, depth, 
                                  num_heads, patch_size, img_size)
        # Translated comment
        self.decoder = UNetDecoder(embed_dim)

        # Translated comment
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # => [B,64,1,1]
            nn.Flatten(),                 # => [B,64]
            nn.Linear(64, 1)             # => [B,1]
        )

    def forward(self, x):
        # Translated comment
        x_enc = self.encoder(x)
        # Translated comment
        x_dec = self.decoder(x_enc)
        # Translated comment
        out = self.cls_head(x_dec)
        return out


# --------------------------------------------------------------------
# Translated comment
# --------------------------------------------------------------------
# if __name__ == "__main__":
#     model = ViTUNetBinaryClassifier(
#         in_chans=27,      # 27 channels
# Translated comment
#         depth=4,          # 4 layer TransformerBlock
#         num_heads=8,
# Translated comment
#         img_size=128      # input 128x128
#     )

#     x = torch.randn(2, 27, 128, 128)  # batch_size=2
#     logits = model(x)                 # => [2,1]
    
#     print(model)
#     print("Input shape :", x.shape)
#     print("Output shape:", logits.shape)
#     print("Logits      :", logits)
