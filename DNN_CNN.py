
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

class DNN_CNN(nn.Module):

    def __init__(self, output_classes):
        super().__init__()

        self.output_classes = output_classes

        self.norm = nn.GroupNorm(num_groups=1, num_channels=5)

        self.conv0 = nn.Sequential(
            # 5@5x28 -> 5@5x28
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3),  stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(5),
            # 5@5x28 -> 5@2x14
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            # 5@2x14 -> 5@2x14
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(5),#
            # 5@2x14 -> 5@1x7
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            # 5@1x7 -> 5@1x7
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(5),#
            # 5@1x7 -> 5@1x3
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        # 5@1x3 -> 5*1*3 = 15
        self.flatten0 = nn.Flatten()

        # 15 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=15, out_features=128),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.8),
            nn.ReLU()
        )
        # 128 -> 72
        self.linear2 = nn.Linear(
            in_features=128, out_features=self.output_classes
        )

        # 512 -> 512
        self.act = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        # Normalise batch
        x = self.norm(input_data)

        # Neural Net
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten0(x)

        x = self.linear0(x)
        x = self.linear1(x)
        predictions = self.linear2(x)

        # predictions = self.softmax(predictions)

        return predictions

