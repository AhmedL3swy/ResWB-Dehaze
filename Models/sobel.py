from env import *
from Models.hazemap import *
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
class Sobel_UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=8):
        super(Sobel_UNet, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet1(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet1(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet1(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet1(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet1(nf * 8, nf * 8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        # dlayer6 = blockUNet1(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        dlayer6 = blockUNet1(nf * 8, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer5 = blockUNet1(nf * 16, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer4 = blockUNet1(nf * 16, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer3 = blockUNet1(nf * 8, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer2 = blockUNet1(nf * 4, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet1(nf * 2, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv = nn.Conv2d(nf * 2, output_nc, 3, padding=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape  
        mod1 = h % 64
        mod2 = w % 64
        if (mod1):
            down1 = 64 - mod1
            x = F.pad(x, (0, 0, 0, down1), "reflect")
        if (mod2):
            down2 = 64 - mod2
            x = F.pad(x, (0, down2, 0, 0), "reflect")
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        dout1 = self.tail_conv(dout1)
        if (mod1): dout1 = dout1[:, :, :-down1, :]
        if (mod2): dout1 = dout1[:, :, :, :-down2]
        return dout1