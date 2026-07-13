import torch.nn as nn
import torch

class FirstConv(nn.Module):
    def __init__(self, alpha, in_channels=3):
        super().__init__()
        out_channels = int(32*alpha)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, downsample=False):
        super().__init__()

        in_channels = int(in_channels * alpha)
        out_channels = int(out_channels * alpha)
        stride = 2 if downsample else 1
        
        self.dwconv = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_channels,
                                bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=in_channels)

        self.pwconv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False
                                )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU6()
        
    def forward(self, x):
        x = self.relu(self.bn0(self.dwconv(x)))
        x = self.relu(self.bn1(self.pwconv(x)))
        return x
        

class ModifiedMobileNetV1(nn.Module):
    def __init__(self, alpha=0.25, num_classes=1, in_channels=3, num_512_blocks=5):
        super().__init__()

        self.first_conv = FirstConv(alpha, in_channels)
        self.depthwise_sep_conv0 = DepthwiseSeparableConv(32, 64, alpha)
        self.depthwise_sep_conv1 = DepthwiseSeparableConv(64, 128, alpha, downsample=True)
        self.depthwise_sep_conv2 = DepthwiseSeparableConv(128, 128, alpha)
        self.depthwise_sep_conv3 = DepthwiseSeparableConv(128, 256, alpha, downsample=True)
        self.depthwise_sep_conv4 = DepthwiseSeparableConv(256, 256, alpha)
        self.depthwise_sep_conv5 = DepthwiseSeparableConv(256, 512, alpha, downsample=True)
        self.depthwise_sep_conv6 = nn.ModuleList(
            [DepthwiseSeparableConv(512, 512, alpha) for _ in range(num_512_blocks)]
        )
        self.depthwise_sep_conv7 = DepthwiseSeparableConv(512, 1024, alpha, downsample=True)
        self.depthwise_sep_conv8 = DepthwiseSeparableConv(1024, 1024, alpha)

        num_output_channels = self.depthwise_sep_conv8.pwconv.out_channels

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(num_output_channels, num_classes)
                                  
    def forward(self, x):
        x = self.first_conv(x)
        
        x = self.depthwise_sep_conv0(x)
        x = self.depthwise_sep_conv1(x)
        x = self.depthwise_sep_conv2(x)
        x = self.depthwise_sep_conv3(x)
        x = self.depthwise_sep_conv4(x)
        x = self.depthwise_sep_conv5(x)
        for layer in self.depthwise_sep_conv6:
            x = layer(x)
        x = self.depthwise_sep_conv7(x)
        x = self.depthwise_sep_conv8(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    model = ModifiedMobileNetV1(alpha=0.25, num_classes=1)
    x = torch.randn(1, 3, 96, 96)
    out = model(x)
    assert out.shape == (1, 1), f"Expected shape (1, 1), got {out.shape}"
    print(f"OK — output shape: {out.shape}")