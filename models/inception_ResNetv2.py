'''
inceptionresnetv2 adapted from: https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/tree/04ca904e8469d685787288f459651e2a8d4ef227/model
and https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py
'''
import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        # stage 1
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=2, padding=0),  # 149 x 149 x 32
            Conv2d(32, 32, 3, stride=1, padding=0),  # 147 x 147 x 32
            Conv2d(32, 64, 3, stride=1, padding=1),  # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2, padding=0),  # 73 x 73 x 64
            Conv2d(64, 80, 1, stride=1, padding=0),  # 73 x 73 x 80
            Conv2d(80, 192, 3, stride=1, padding=0),  # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2, padding=0),  # 35 x 35 x 192
        )

        self.branch_0 = Conv2d(192, 96, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv2d(192, 48, 1, stride=1, padding=0),
            Conv2d(48, 64, 5, stride=1, padding=2),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(192, 64, 1, stride=1, padding=0),
            Conv2d(64, 96, 3, stride=1, padding=1),
            Conv2d(96, 96, 3, stride=1, padding=1),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(192, 64, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0),
            Conv2d(32, 32, 3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0),
            Conv2d(32, 48, 3, stride=1, padding=1),
            Conv2d(48, 64, 3, stride=1, padding=1)
        )
        # bias=True
        self.conv = nn.Conv2d(128, 320, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 3, stride=2, padding=0)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0),
            Conv2d(256, 256, 3, stride=1, padding=1),
            Conv2d(256, 384, 3, stride=2, padding=0),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 128, 1, stride=1, padding=0),
            Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3)),
            Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0))
        )
        # bias=True
        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0),
            Conv2d(256, 384, 3, stride=2, padding=0)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0),
            Conv2d(256, 288, 3, stride=2, padding=0),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0),
            Conv2d(256, 288, 3, stride=1, padding=1),
            Conv2d(288, 320, 3, stride=2, padding=0)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0),
            Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1)),
            Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0))
        )
        # bias=True
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=1000):
        super(Inception_ResNetv2, self).__init__()
        blocks = [Stem(in_channels)]
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_average_pooling = nn.AvgPool2d(8, count_include_pad=False)
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
