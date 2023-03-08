
from env import *
from Models.restormer import *
from Models.deep_wb import *
from Models.hazemap import *
from Models.discriminator import *
from Models.enhancer import *
from Models.gblock import *
from Models.sobel import *
class Custom_fusion_net(nn.Module):
    def __init__(self):
        super(Custom_fusion_net, self).__init__()
        self.restormer = Restormer()
        checkpoint = torch.load("/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/motion_deblurring.pth")
        self.restormer.load_state_dict(checkpoint['params'])
        self.sobel_UNet = Sobel_UNet()
        self.haze_density = HazeDensityMap()
        self.haze_density.load_state_dict(torch.load("/content/drive/MyDrive/Graduation Project/HM.pt"))
        self.GBlock = GBlock(3,3)
        self.awb =  deepWBNet()
        checkpoints = torch.load("/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/net_awb.pth")
        self.awb.load_state_dict(checkpoints['state_dict'])
    def forward(self, input):
        restormer=self.restormer(input)
        x = self.haze_density(input)
        # dwt_branch = self.dwt_branch(input)
        hazy_sobel = self.sobel_UNet(restormer)
        x = self.GBlock(x,hazy_sobel,restormer)
        x = self.awb(x)    
        return x , hazy_sobel
    