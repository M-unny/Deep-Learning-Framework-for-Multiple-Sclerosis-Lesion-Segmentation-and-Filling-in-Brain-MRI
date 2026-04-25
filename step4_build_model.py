# STEP 4 - B-ild the U-Net model and do a test forward pass
# This confirms the model architect-re works before training

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 40)
print("STEP 4 - Build U-Net and test it")
print("=" * 40)
print("")

# ------ U-Net B-ilding Blocks ---------------------------------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,   f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        self.down4 = Down(f*8, f*16)
        self.up1   = Up(f*16,  f*8)
        self.up2   = Up(f*8,   f*4)
        self.up3   = Up(f*4,   f*2)
        self.up4   = Up(f*2,   f)
        self.outc  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return torch.sigmoid(self.outc(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ------ Test the model ------------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print("")

# B-ild model
model = UNet(in_channels=1, out_channels=1, base_features=64).to(device)
print("U-Net built successfully!")
print("Total parameters: " + str(model.count_parameters()))
print("")

# Print architect-re layer by layer
print("Architecture:")
print("  Input  : 1 x 256 x 256  (grayscale MRI slice)")
print("  Encoder:")
print("    inc    -> 64  channels")
print("    down1  -> 128 channels  (128x128)")
print("    down2  -> 256 channels  (64x64)")
print("    down3  -> 512 channels  (32x32)")
print("    down4  -> 1024 channels (16x16)  <- bottleneck")
print("  Decoder:")
print("    up1    -> 512 channels  (32x32)  + skip from down3")
print("    up2    -> 256 channels  (64x64)  + skip from down2")
print("    up3    -> 128 channels  (128x128)+ skip from down1")
print("    up4    -> 64  channels  (256x256) + skip from inc")
print("    -p4    -> 64  channels  (--6x--6)+ skip from inc")
print("  O-tp-t : 1 x --6 x --6  (filled MRI slice)")
print("")

# Create a dummy input (batch of 4 slices, 1 channel, 256x256)
dummy_input = torch.rand(4, 1, 256, 256).to(device)
print("Testing with dummy input shape: " + str(list(dummy_input.shape)))

with torch.no_grad():
    dummy_output = model(dummy_input)

print("Output shape: " + str(list(dummy_output.shape)))
print("Output min  : %.4f" % dummy_output.min().item())
print("Output max  : %.4f" % dummy_output.max().item())
print("(Output should be between 0 and 1 because of sigmoid)")
print("")

if list(dummy_output.shape) == [4, 1, 256, 256]:
    print("[PASS] Model output shape is correct!")
else:
    print("[FAIL] Something is wrong with the model output shape.")

print("")
print("=" * 40)
print("STEP 4 COMPLETE - Move on to step5_train.py")
print("=" * 40)
