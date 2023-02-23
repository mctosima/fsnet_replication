import torch
import torch.nn as nn
from torchinfo import summary

"""
A replication of FS-Net network architecture
Paper: https://ieeexplore.ieee.org/document/9751737
"""


class FNet(nn.Module):
    def __init__(
        self, input_size: int = 7, hidden_size: int = 32, output_size: int = 16, num_res_blocks: int = 3,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)

        # 3 layers of FC residual blocks
        self.fc_res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.fc_res_blocks.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ))
            self.relu_residual = nn.ReLU()

        # fc with 16 output size
        self.fc5 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
        )

        # fc with 1 output size to auxiliar the loss function
        self.fc6 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)

        # 3 layers of FC residual blocks (no info about the activation function, I used relu)
        for fc_res_block in self.fc_res_blocks:
            residual = x
            x = fc_res_block(x)
            x += residual
            x = self.relu_residual(x)

        # fc with 16 output size
        x_to_concat = self.fc5(x)

        # fc with 1 output size to auxiliar the loss function
        x_to_loss = self.fc6(x)

        return (x_to_concat, x_to_loss)

class SNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv_kernel: int = 1,
        conv_stride: int = 3,
        conv_padding: int = 0,
        res_kernel: int = 3,
        res_stride: int = 1,
        res_padding: int = 1,
        num_res_blocks: int = 6,
    ):
        super().__init__()

        # conv2d layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(conv_kernel,in_channels), stride=conv_stride, padding=conv_padding)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        # 6x residual block
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=res_kernel,
                    stride=res_stride,
                    padding=res_padding,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=res_kernel,
                    stride=res_stride,
                    padding=res_padding,
                ),
            ))
            self.residual_relu = nn.ReLU(inplace=True)
        

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
        x = self.batchnorm1(x)
        x = torch.relu(x)

        # 6x residual block (no info about the activation function, I used relu)
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x += residual
            x = self.residual_relu(x)

        # adaptive average pooling from 64x600 to 64x3
        x = self.adaptive_avg_pool(x)

        x = x.view(x.size(0), -1)

        # fc with 64x1 output size
        x = self.fc1(x)
        x_to_concat = self.batchnorm8(x)

        # fc for auxiliar the loss function
        x_to_loss = self.fc2(x_to_concat)

        return (x_to_concat, x_to_loss)
    
class FSnet(nn.Module):
    def __init__(
        self,
        f_input_size: int = 7,
        f_hidden_size: int = 32,
        f_output_size: int = 16,
        s_in_channels: int = 3,
        s_out_channels: int = 64,
        s_conv_kernel: int = 1,
        s_conv_stride: int = 3,
        s_conv_padding: int = 0,
        s_res_kernel: int = 3,
        s_res_stride: int = 1,
        s_res_padding: int = 1,
        s_num_res_blocks: int = 6,
    ):
        super().__init__()

        self.fnet = FNet(input_size=f_input_size, hidden_size=f_hidden_size, output_size=f_output_size)
        self.snet = SNet(in_channels=s_in_channels,
                         out_channels=s_out_channels,
                         conv_kernel=s_conv_kernel,
                         conv_stride=s_conv_stride,
                         conv_padding=s_conv_padding,
                         res_kernel=s_res_kernel,
                         res_stride=s_res_stride,
                         res_padding=s_res_padding,
                         num_res_blocks=s_num_res_blocks)
        
        self.fc1 = nn.Linear(80, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, f_input, s_input):
        f_out, f_loss = self.fnet(f_input)
        s_out, s_loss = self.snet(s_input)

        fs_cat = torch.cat((f_out, s_out), dim=1)
        fs_fc1 = self.fc1(fs_cat)
        out = self.fc2(fs_fc1)

        return f_out, s_out, out
        # to compute loss: f_loss + s_loss + out before loss.backdward()


    

if __name__ == "__main__":
    # model = FNet()
    # x = torch.randn(1, 7)
    # summary(model, input_data=(x,))

    # model = SNet()
    # x = torch.randn(1, 3, 1, 1800)
    # summary(model, input_data=(x,))

    model = FSnet(
        f_input_size=7,
        f_hidden_size=32,
        f_output_size=16,
        s_in_channels=3,
        s_out_channels=64,
        s_conv_kernel=1,
        s_conv_stride=3,
        s_conv_padding=0,
        s_res_kernel=3,
        s_res_stride=1,
        s_res_padding=1,
        s_num_res_blocks=6,
    )
    f_input = torch.randn(1, 7)
    s_input = torch.randn(1, 3, 1, 1800)
    summary(model, input_data=[f_input, s_input], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], row_settings=["var_names"])
