import torch
import torch.nn as nn
import torch.nn.functional as F
from Binarization import BinarizeConv2d, BinaryHardtanh

class sConv2d(nn.Module):
    def __init__(self, *argv, **argn):
        super(sConv2d, self).__init__()
        if 'factor' in argn:
            self.init_factor = argn['factor']
            del argn['factor']
        else:
            self.init_factor = 1
        if 'max_scales' in argn:
            self.max_scales = argn['max_scales']
            del argn['max_scales']
        else:
            self.max_scales = 4
        if 'usf' in argn:
            self.usf = argn['usf']
            del argn['usf']
        else:
            self.usf = False
        if 'bnorm' in argn:
            self.bnorm = argn['bnorm']
            del argn['bnorm']
        else:
            self.bnorm = True
        if 'lf' in argn:
            self.lf = argn['lf']
            del argn['lf']
        else:
            self.lf = False
        if 'binarized' in argn:
            self.binarized = argn['binarized']
            del argn['binarized']
        else:
            self.binarized = False
        if 'last' in argn:
            self.last = argn['last']
            del argn['last']
        else:
            self.last = False

        self.alpha = nn.Parameter(torch.ones(self.max_scales))  # *0.001)
        if self.lf:
            self.factor = nn.Parameter(torch.ones(1) * self.init_factor)
        else:
            self.factor = self.init_factor
        self.interp = F.interpolate
        if self.binarized:
            self.conv = nn.ModuleList([BinarizeConv2d(*argv, **argn, bias=True) for _ in range(1)])
            if self.bnorm:
                self.bn = nn.ModuleList([nn.BatchNorm2d(argv[1]) for _ in range(self.max_scales)])
            self.relu = nn.ModuleList([nn.ReLU() for _ in range(self.max_scales)])
        else:
            self.conv = nn.ModuleList([nn.Conv2d(*argv, **argn, bias=True) for _ in range(self.max_scales)])
            if self.bnorm:
                self.bn = nn.ModuleList([nn.BatchNorm2d(argv[1]) for _ in range(self.max_scales)])
            self.relu = nn.ModuleList([nn.ReLU() for _ in range(self.max_scales)])
        self.lastbn = nn.BatchNorm2d(argv[1])
        if not self.last:
            self.hardtanh = BinaryHardtanh()


    def forward(self, x):
        factor = self.factor
        lx = [x]
        ly = []
        self.nres = self.max_scales
        self.nalpha = F.softmax(self.alpha[:self.nres] * factor, 0)
        lcorr = []
        for idx in range(self.nres):
            lx.append((F.max_pool2d(lx[idx], kernel_size=2)))
            y = self.conv[0](lx[idx])
            if self.bnorm:
                y = self.bn[idx](y)
            if self.max_scales != 1:
                y = self.relu[idx](y)

            lcorr.append(idx)
            ly.append(self.interp(y, scale_factor=2 ** idx, mode='nearest'))
        y = (torch.stack(ly, 0))
        ys = y.shape
        sel = F.softmax(self.alpha[lcorr] * factor, 0).view(-1, 1)
        out = (y * sel.view(ys[0], 1, 1, 1, 1)).sum(0)
        out = self.lastbn(out)
        if not self.last:
            out = self.hardtanh(out)
        return out
