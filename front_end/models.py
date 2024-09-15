import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Convolutional block consisting of Sequential operations:
      - 3x3 Conv
      - BatchNorm
      - ReLU
      - 3x3 Conv
      - BatchNorm
      - ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class RecurrentBlock(nn.Module):
    """
    Recurrent convolutional block consisting of:
      - 3x3 Conv
      - ReLU
      - 3x3 Conv
      - ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention block to compute attention between 
    encoder and decoder features.
    """
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        
        self.theta = nn.Conv3d(encoder_channels, encoder_channels // 2, kernel_size=1)
        self.phi = nn.Conv3d(decoder_channels, decoder_channels // 2, kernel_size=1)
        self.psi = nn.Conv3d(encoder_channels, encoder_channels // 2, kernel_size=1)
        self.out_conv = nn.Conv3d(encoder_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, encoder_x, decoder_x):
        theta = self.theta(encoder_x) 
        phi = self.phi(decoder_x)
        psi = self.psi(encoder_x)
        attn = self.relu(theta + phi + psi)
        attn = self.out_conv(attn)
        attn = self.sigmoid(attn)
        return encoder_x * attn.expand_as(encoder_x) + decoder_x



class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, window_size=4):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = in_channels * window_size * window_size * window_size
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.window_size = window_size

        # Ensure embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Split input into non-overlapping windows and flatten
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // self.window_size, self.window_size, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B, D // self.window_size * H // self.window_size * W // self.window_size, C * self.window_size * self.window_size * self.window_size)

        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        # Apply MLP
        x = x + self.mlp(x)

        # Reshape back to original shape
        x = x.view(B, D // self.window_size, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.window_size, C).permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, D, H, W)

        # Transform channels to desired output channels
        x = self.conv1x1x1(x)
        return x

class SAD_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SAD_UNet, self).__init__()

        # Encoder with Swin Transformer blocks
        self.enc1 = SwinTransformerBlock(in_channels = in_channels, out_channels= 64, window_size=2)
        self.enc2 = SwinTransformerBlock(64, out_channels= 128, window_size=2)
        self.enc3 = SwinTransformerBlock(128, out_channels= 256, window_size=2)
        self.enc4 = SwinTransformerBlock(256, out_channels= 512, window_size=2)
        self.enc5 = SwinTransformerBlock(512, out_channels= 1024, window_size=2)


        # Recurrent Block
        self.recurrent_block = RecurrentBlock(1024, 256)

        # Decoder with Attention blocks
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)

        self.dec1 = ConvBlock(512 + 512, 512)
        self.dec2 = ConvBlock(256 + 256, 256)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.dec4 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Deep supervision classifiers
        self.aux_classifier1 = nn.Conv3d(512, num_classes, kernel_size=1)
        self.aux_classifier2 = nn.Conv3d(256, num_classes, kernel_size=1)
        self.aux_classifier3 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.aux_classifier4 = nn.Conv3d(64, num_classes, kernel_size=1)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = F.max_pool3d(e1, 2)
        e2 = self.enc2(x)
        x = F.max_pool3d(e2, 2)
        e3 = self.enc3(x)
        x = F.max_pool3d(e3, 2)
        e4 = self.enc4(x)
        x = F.max_pool3d(e4, 2)
        x = self.enc5(x)
        x = self.recurrent_block(x)

        aux_output1 = self.aux_classifier1(e4)
        x = self.upconv1(x)
        x = self.att1(e4, x)
        x = self.dec1(torch.cat((e4, x), dim=1))

        aux_output2 = self.aux_classifier2(e3)
        x = self.upconv2(x)
        x = self.att2(e3, x)
        x = self.dec2(torch.cat((e3, x), dim=1))

        aux_output3 = self.aux_classifier3(e2)
        x = self.upconv3(x)
        x = self.att3(e2, x)
        x = self.dec3(torch.cat((e2, x), dim=1))

        aux_output4 = self.aux_classifier4(e1)
        x = self.upconv4(x)
        x = self.att4(e1, x)
        x = self.dec4(torch.cat((e1, x), dim=1))

        # Resize auxiliary outputs to match target size
        aux_output1 = F.interpolate(aux_output1, size=(128,128,128))
        aux_output2 = F.interpolate(aux_output2, size=(128,128,128))
        aux_output3 = F.interpolate(aux_output3, size=(128,128,128))
        aux_output4 = F.interpolate(aux_output4, size=(128,128,128))

        x = self.final_conv(x)
        return x, aux_output1, aux_output2, aux_output3, aux_output4
