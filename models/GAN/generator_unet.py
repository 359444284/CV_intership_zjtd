# adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet
# pad and unpad adapted_from https : //github.com/seoungwugoh/STM/blob/905f11492a6692dd0d0fa395881a8ec09b211a36/helpers.py#L33
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(2e-1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(2e-1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        factor = 2 if bilinear else 1
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128 // factor)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Generator_UNet(nn.Module):
    def __init__(self, img_size_min, num_scale, scale_factor=4/3):
        super(Generator_UNet, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.current_scale = 0

        self.size_list = [round(self.img_size_min * scale_factor**i) for i in range(num_scale + 1)]
        print(self.size_list)

        self.sub_generators = nn.ModuleList()
        first_generator = UNet(3, 3, True)
        self.sub_generators.append(first_generator)

    def forward(self, z, img=None):
        x_list = []
        x_pad, pads = pad_to(z[0], 8)
        x_first = self.sub_generators[0](x_pad)
        x_unpad = unpad(x_first, pads)
        x_list.append(x_unpad)

        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1):
            x_inter = F.interpolate(x_inter, (self.size_list[i], self.size_list[i]), mode='bilinear', align_corners=True)
            x_prev = x_inter
            x_inter = x_inter + z[i]
            x_pad, pads = pad_to(x_inter, 8)
            x_pad = self.sub_generators[i](x_pad)
            x_inter = unpad(x_pad, pads) + x_prev

            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        tmp_generator = UNet(3, 3, True)

        # if self.current_scale %8 != 0:
        prev_generator = self.sub_generators[-1]

        # Initialize layers via copy
        if self.current_scale >= 1:
            tmp_generator.load_state_dict(prev_generator.state_dict())

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")