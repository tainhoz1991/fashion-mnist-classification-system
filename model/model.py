import torch.nn as nn
import torch.nn.functional as F
from base import ModelBase


class FashionMNISTModel(ModelBase):
    def __init__(self, hidden_size, output_size=10):
        super(FashionMNISTModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size * 7 * 7, out_features=output_size),
        )

    def forward(self, x):
        x = self.classifier(self.conv_block_2(self.conv_block_1(x)))
        return F.log_softmax(x, dim=1)
