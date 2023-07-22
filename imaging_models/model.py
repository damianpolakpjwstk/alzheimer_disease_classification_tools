"""CNN Network for MRI classification."""

import torch
import torchsummary
from torch import nn

from pseudo3d_pytorch.src.blocks import P3DBlockTypeA, P3DBlockTypeB, P3DBlockTypeC, AttentionBlock3D


class MultiScaleStem(nn.Module):
    """
    Multiscale stem block for MRI classification. It consists of three Pseudo-3D ResNet-like blocks with different
    dilation rates (1, 2, 3) and a max pooling layer. The output of the block is concatenated in filter dimension
    and passed through a 1x1x1 convolution to reduce the number of channels.

    It is inspired by the network from "Multi-scale attention-based pseudo-3D convolution neural network
    for Alzheimer’s disease diagnosis using structural MRI" paper (https://doi.org/10.1016/j.patcog.2022.108825).
    Instead of using transposed convolutions, it uses dilation to achieve similar effect.
    """

    def __init__(self, num_filters: int = 12, pad_size: int = 5, block_type: str = "A",
                 base_channels: int = 32) -> None:
        """
        Initialize the multiscale stem block.

        :param num_filters: number of output filters in the first convolutional layers of each block.
        :param pad_size: size of the padding in the max pooling layer. It is used to reduce the size of the input.
        :param block_type: type of the Pseudo-3D ResNet-like block. It can be "A", "B" or "C".
        :param base_channels: number of output channels of the 1x1x1 convolution.
        """
        super().__init__()
        assert block_type in ("A", "B", "C"), "Block type must be one of the following: A, B, C"
        block = P3DBlockTypeA if block_type == "A" else P3DBlockTypeB if block_type == "B" else P3DBlockTypeC

        self.small_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.medium_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=2),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.large_scale_stem = nn.Sequential(
            block(1, num_filters, num_filters, kernel_size=3, stride=2, dilation=3),
            nn.MaxPool3d(kernel_size=pad_size, stride=2, padding=1)
        )
        self.downsample = nn.Conv3d(num_filters * 3, base_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multiscale stem block."""
        x_small = self.small_scale_stem(x)
        x_medium = self.medium_scale_stem(x)
        x_large = self.large_scale_stem(x)
        x = torch.cat((x_small, x_medium, x_large), dim=1)
        x = self.downsample(x)
        return x


class MRINet(nn.Module):
    """
    Network for MRI classification.
    """
    def __init__(self, num_classes: int = 2, dropout_value: float | None = 0.2,
                 use_multiscale_stem: bool = False, base_channels: int = 32, pretrain: bool = False) -> None:
        """
        Initialize the network.

        :param num_classes: number of classes in the classification task.
        :param dropout_value: dropout value for the last fully connected layer.
        :param use_multiscale_stem: whether to use multiscale stem block or not. If False, the network uses
        Pseudo-3D ResNet-like blocks in the stem with MaxPool3d with kernel_size=5 and stride=2.
        :param base_channels: number of output channels of the 1x1x1 convolution in the multiscale stem block.
        :param pretrain: Set to True to create classificator for pretraining. It uses two fully connected layers
        instead of one and returns two outputs.
        """
        super().__init__()
        self.base_channels = base_channels
        self.pretrain = pretrain

        self.stem = nn.Sequential(
            P3DBlockTypeA(1, 24, self.base_channels, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=5, stride=2, padding=0)
        ) if not use_multiscale_stem else \
            MultiScaleStem(num_filters=12, pad_size=4, block_type="A", base_channels=self.base_channels)

        self.attention_block_0 = AttentionBlock3D(num_filters=self.base_channels, reduction_ratio=4)

        self.block1 = nn.Sequential(
            P3DBlockTypeA(self.base_channels, 16, 64, kernel_size=3, stride=2, dilation=1),
            AttentionBlock3D(num_filters=64, reduction_ratio=4),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0)
        )

        self.block2 = nn.Sequential(
            P3DBlockTypeA(64, 32, 128, kernel_size=3, stride=2, dilation=1),
            AttentionBlock3D(num_filters=128, reduction_ratio=4),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0)
        )

        self.block3 = nn.Sequential(
            P3DBlockTypeA(128, 64, 256, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),
        )

        self.global_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_value) if dropout_value is not None else nn.Identity(),
                nn.Linear(256 * 1 * 1 * 1, num_classes),
            )

        if self.pretrain:
            self.pretrain_classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_value) if dropout_value is not None else nn.Identity(),
            )
            self.output_0 = nn.Linear(256 * 1 * 1 * 1, 1)
            self.output_1 = nn.Linear(256 * 1 * 1 * 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network."""
        x = self.stem(x)
        x = self.attention_block_0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        if self.pretrain:
            x = self.pretrain_classifier(x)
            output_0 = self.output_0(x)
            output_1 = self.output_1(x)
            return output_0, output_1
        x = self.classifier(x)
        return x

    def freeze(self):
        """Freeze all layers except the last fully connected layer."""
        print("Freezing all layers except the last fully connected layer.")
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all layers."""
        print("Unfreezing all layers.")
        for name, param in self.named_parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = MRINet(num_classes=2).to("cuda")
    print(torchsummary.summary(model, (1, 224, 224, 224)))
