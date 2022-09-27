
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt


class DNN_max(nn.Module):

    def __init__(self, output_classes):
        super().__init__()

        self.flatten0 = nn.Flatten()

        # 280
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=10, out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=896),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=896, out_features=512),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 1024
        self.linear4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # 1024 -> 72
        self.linear5 = nn.Linear(
            in_features=128, out_features=output_classes
        )

        # 512 -> 512
        self.act = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        x = self.flatten0(input_data)

        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        # x = self.softmax(x)

        return x

