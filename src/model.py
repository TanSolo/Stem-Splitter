import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def center_crop(x, target_h, target_w):
    _, _, h, w = x.shape
    if h == target_h and w == target_w:
        return x
    dh = h - target_h
    dw = w - target_w
    if dh < 0 or dw < 0:
        target_h = min(h, target_h)
        target_w = min(w, target_w)
        dh = h - target_h
        dw = w - target_w
    top = dh // 2
    left = dw // 2
    return x[:, :, top:top + target_h, left:left + target_w]


class UNetMask(nn.Module):
    """
    Input: magnitude spectrogram [B, 2, F, TT] (stereo treated as channels)
    Output: masks [B, stems, 2, F, TT]
    """
    def __init__(self, base=32, stems=4):
        super().__init__()
        self.stems = stems

        self.enc1 = ConvBlock(2, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, stems * 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        e3c = center_crop(e3, d3.size(2), d3.size(3))
        d3 = self.dec3(torch.cat([d3, e3c], dim=1))

        d2 = self.up2(d3)
        e2c = center_crop(e2, d2.size(2), d2.size(3))
        d2 = self.dec2(torch.cat([d2, e2c], dim=1))

        d1 = self.up1(d2)
        e1c = center_crop(e1, d1.size(2), d1.size(3))
        d1 = self.dec1(torch.cat([d1, e1c], dim=1))

        masks = torch.sigmoid(self.out(d1))  # [B, stems*2, F, TT]
        B, C, F, TT = masks.shape
        masks = masks.view(B, self.stems, 2, F, TT)
        return masks
