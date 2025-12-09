import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Depthwise Separable Convolution (경량화 핵심)
# ============================================================
class DSConv(nn.Module):
    """
    일반 Conv2d보다 파라미터 수와 연산량이 훨씬 적은 블록입니다.
    MobileNet 등 모바일 모델에서 주로 사용합니다.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. Depthwise: 채널별로 따로따로 3x3 합성곱
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        # 2. Pointwise: 1x1 합성곱으로 채널 섞기
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_ch) # 모바일 모델은 학습 안정성을 위해 BN 추천
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

# U-Net용 이중 Conv 블록 (DSConv로 교체)
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        DSConv(in_ch, out_ch),
        DSConv(out_ch, out_ch)
    )

# ============================================================
# 2. FiLM Layer (가볍게 유지)
# ============================================================
class FiLM(nn.Module):
    def __init__(self, channels, user_dim=4, hidden=16): # hidden도 32->16으로 축소
        super().__init__()
        self.fc1 = nn.Linear(user_dim, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, feat, user_vec):
        h = torch.relu(self.fc1(user_vec))
        gamma, beta = self.fc2(h).chunk(2, dim=1)
        
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return feat * gamma + beta

# ============================================================
# 3. Mobile FiLM U-Net (전체 모델)
# ============================================================
class FiLM_UNet(nn.Module):
    def __init__(self, user_dim=4, base=16): # ★ base를 32에서 16으로 줄임 (크기 대폭 감소)
        super().__init__()

        # Encoder
        self.e1 = conv_block(3, base)
        self.p1 = nn.MaxPool2d(2)

        self.e2 = conv_block(base, base*2)
        self.p2 = nn.MaxPool2d(2)

        self.e3 = conv_block(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base*4, base*8)
        self.film = FiLM(base*8, user_dim=user_dim, hidden=base) # FiLM도 base 크기에 맞춤

        # Decoder
        # ConvTranspose2d는 무거우니 Upsample + Conv로 대체할 수도 있지만, 
        # 일단 성능 유지를 위해 유지하되 채널은 줄어든 base를 따름
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = conv_block(base*2, base)

        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x, user_vec):
        # Encoder
        e1 = self.e1(x)
        p1 = self.p1(e1)

        e2 = self.e2(p1)
        p2 = self.p2(e2)

        e3 = self.e3(p2)
        p3 = self.p3(e3)

        # Bottleneck + FiLM
        b = self.bottleneck(p3)
        b = self.film(b, user_vec)

        # Decoder
        u3 = self.up3(b)
        d3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(d1))