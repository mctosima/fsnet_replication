import torch
import torch.nn as nn
from torchinfo import summary

"""
A replication of FS-Net network architecture
Paper: https://ieeexplore.ieee.org/document/9751737
"""


class FNet(nn.Module):
    def __init__(
        self, input_size: int = 7, hidden_size: int = 32, output_size: int = 16
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)

        # 3 layers of FC residual blocks
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.batchnorm4 = nn.BatchNorm1d(hidden_size)

        # fc with 16 output size
        self.fc5 = nn.Linear(hidden_size, output_size)

        # fc with 1 output size to auxiliar the loss function
        self.fc6 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.fc1(x)))

        # 3 layers of FC residual blocks (no info about the activation function, I used relu)
        x = torch.relu(self.batchnorm2(self.fc2(x)) + x)
        x = torch.relu(self.batchnorm3(self.fc3(x)) + x)
        x = torch.relu(self.batchnorm4(self.fc4(x)) + x)

        # fc with 16 output size
        x_to_concat = self.fc5(x)

        # fc with 1 output size to auxiliar the loss function
        x_to_loss = self.fc6(x)

        return x_to_concat, x_to_loss


class SNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv_kernel: int = (1,3),
        conv_stride: int = 3,
        conv_padding: int = 0,
        res_kernel: int = 3,
        res_stride: int = 1,
        res_padding: int = 1,
    ):
        super().__init__()

        # conv2d layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        # 6x residual block
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm5 = nn.BatchNorm2d(out_channels)
        self.conv6 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm6 = nn.BatchNorm2d(out_channels)
        self.conv7 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=res_kernel,
            stride=res_stride,
            padding=res_padding,
        )
        self.batchnorm7 = nn.BatchNorm2d(out_channels)

        # adaptive average pooling from 64x600 to 64x3
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 3))

        # fc with 64x1 output size
        self.fc1 = nn.Linear(64 * 3, 64)
        # batch norm
        self.batchnorm8 = nn.BatchNorm1d(64)

        # fc for auxiliar the loss function
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        print(f"conv1: {x.shape}")
        x = self.batchnorm1(x)
        print(f"batchnorm1: {x.shape}")
        x = torch.relu(x)
        print(f"relu: {x.shape}")

        # 6x residual block (no info about the activation function, I used relu)
        x = torch.relu(self.batchnorm2(self.conv2(x)) + x)
        print(f"conv2: {x.shape}")
        x = torch.relu(self.batchnorm3(self.conv3(x)) + x)
        x = torch.relu(self.batchnorm4(self.conv4(x)) + x)
        x = torch.relu(self.batchnorm5(self.conv5(x)) + x)
        x = torch.relu(self.batchnorm6(self.conv6(x)) + x)
        x = torch.relu(self.batchnorm7(self.conv7(x)) + x)
        print(f"conv7: {x.shape}")

        # adaptive average pooling from 64x600 to 64x3
        x = self.adaptive_avg_pool(x)
        print(f"adaptive_avg_pool: {x.shape}")

        x = x.view(x.size(0), -1)
        print(f"view: {x.shape}")

        # fc with 64x1 output size
        x = self.fc1(x)
        x_to_concat = self.batchnorm8(x)
        print(f"fc1: {x.shape}")

        # fc for auxiliar the loss function
        x_to_loss = self.fc2(x_to_concat)
        print(f"fc2: {x_to_loss.shape}")

        return x_to_concat, x_to_loss
    

if __name__ == "__main__":
    # model = FNet()
    # x = torch.randn(1, 7)
    # summary(model, input_data=(x,))

    model = SNet()
    x = torch.randn(1, 3, 1, 1800)
    summary(model, input_data=(x,))
