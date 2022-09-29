
from torch import nn


class DNN_CNN_Chakrabarty(nn.Module):

    def __init__(self, output_classes):
        super().__init__()

        self.output_classes = output_classes

        self.conv0 = nn.Sequential(
            # 1@5x129 -> 64@5x128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2),  stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        self.conv1 = nn.Sequential(
            # 64@5x128 -> 64@5x127
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        self.conv2 = nn.Sequential(
            # 64@5x127 -> 64@5x126
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        # 64@5x126 -> 64*5*126 = 40320
        self.flatten0 = nn.Flatten()

        # 15 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=40320, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # 128 -> 72
        self.linear2 = nn.Linear(
            in_features=512, out_features=self.output_classes
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        # Neural Net
        x = self.conv0(input_data)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten0(x)

        x = self.linear0(x)
        x = self.linear1(x)
        predictions = self.linear2(x)

        predictions = self.softmax(predictions)

        return predictions

