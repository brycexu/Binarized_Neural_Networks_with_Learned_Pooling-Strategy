from MuLayer import sConv2d
import torch.nn as nn


class MutipleBNN(nn.Module):
    def __init__(self):
        super(MutipleBNN, self).__init__()
        self.convolutions = nn.Sequential(

            sConv2d(3, 32, kernel_size=3, stride=1, padding=1, max_scales=1,
                    usf=True, binarized=False),

            sConv2d(32, 32, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(32, 64, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(64, 64, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(64, 128, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(128, 128, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(128, 256, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(256, 256, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(256, 512, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=True),

            sConv2d(512, 10, kernel_size=3, stride=1, padding=1, max_scales=4,
                    usf=True, binarized=False, last=True),

            nn.AvgPool2d(kernel_size=32)

        )
        self.softmax = nn.Sequential(

            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 10 * 1 * 1)
        out = self.softmax(x)
        return out