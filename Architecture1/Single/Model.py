from Binarization import BinarizeConv2d, BinarizeLinear, BinaryHardtanh
import torch.nn as nn

class SingleBNN(nn.Module):
    def __init__(self):
        super(SingleBNN, self).__init__()
        self.convolutions = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            BinaryHardtanh(),

            BinarizeConv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            BinaryHardtanh(),

            BinarizeConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            BinaryHardtanh(),

            BinarizeConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            BinaryHardtanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            BinaryHardtanh(),

            BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            BinaryHardtanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinarizeConv2d(256, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.AvgPool2d(kernel_size=8)

        )
        self.softmax = nn.Sequential(

            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 10)
        out = self.softmax(x)
        return out