
from torch import nn


class DNN_CNN_Chakrabarty(nn.Module):

    def __init__(self, output_classes):
        super().__init__()

        self.output_classes = output_classes

        self.norm = nn.GroupNorm(num_groups=1, num_channels=1)

        self.conv0 = nn.Sequential(
            # 1@5x129 -> 64@4x128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 2),  stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            # 64@4x128 -> 64@3x127
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            # 64@3x127 -> 64@2x126
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        # 64@2x126 -> 64*2*126 = 16128
        self.flatten0 = nn.Flatten()

        # 16128 -> 512
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=16128, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # 512 -> 512
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # 512 -> 72
        self.linear2 = nn.Linear(
            in_features=512, out_features=self.output_classes
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        input_data = self.norm(input_data)

        # Neural Net
        x = self.conv0(input_data)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten0(x)

        x = self.linear0(x)
        x = self.linear1(x)
        predictions = self.linear2(x)

        #predictions = self.softmax(predictions)

        return predictions

