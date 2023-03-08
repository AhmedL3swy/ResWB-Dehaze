from env import *

class GBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(GBlock, self).__init__()

        self.c = nn.Conv2d(in_c + in_c, out_c, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2], axis = 1)
        x = self.c(x)
        x = self.sig(x)
        x = x * x3
        return x