from env import *
from utils import *
from custom_fusion import *
from dataloader import *
from losses import *
from Models.deep_wb import *
from Models.discriminator import *
from Models.enhancer import *
from Models.gblock import *
from Models.restormer import *
from Models.sobel import *
from Models.hazemap import *
# VAL_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/valid_dense/HAZY/"
# VAL_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/valid_dense/GT/"
VAL_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/Test_Hazy/"
VAL_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/Test_GT/"

VAL_BATCH_SIZE = 1
NUM_WORKERS = 0

# --- output picture and check point --- #
G_model_save_dir = "/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/Generator_NH_Restormer_HM_NAFNET_GBlock_Enhancer_Sobel.pth"
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Custom_fusion_net().float()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)

# --- Load testing data --- #
val_data = CustomDataLoader(HAZY_path = VAL_HAZY_IMAGES_PATH,
                            GT_path = VAL_GT_IMAGES_PATH,
                            image_size = (1152,1600),
                            white_balance = False,
                            crop = False,
                            resize = False)

val_loader = DataLoader(val_data, 
                        batch_size = VAL_BATCH_SIZE, 
                        num_workers = NUM_WORKERS)

MyEnsembleNet = MyEnsembleNet.to(device)
val_data = dehaze_test_dataset(VAL_HAZY_IMAGES_PATH, VAL_GT_IMAGES_PATH)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0)
# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(G_model_save_dir))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')
# --- Start training --- #
print("-----Testing-----")
with torch.inference_mode():
    psnr_list = []
    ssim_list = []
    MyEnsembleNet.eval()
    for batch_idx, (frame1, frame2,frame3,frame4,name,clean) in enumerate(val_loader):
            frame1 = frame1.to(device)
            frame2 = frame2.to(device)
            frame3 = frame3.to(device)
            frame4 = frame4.to(device)
            clean = clean.to(device)
            #frame_out_up,_ = MyEnsembleNet(hazy_up)
            #frame_out_down,_ = MyEnsembleNet(hazy_down)
            frameo1,_ = MyEnsembleNet(frame1)
            frameo2,_ = MyEnsembleNet(frame2)
            frameo3,_ = MyEnsembleNet(frame3)
            frameo4,_ = MyEnsembleNet(frame4)

            frame_out = (torch.cat([frameo1.permute(0, 2, 3, 1), frameo2[:, :, 20:, :].permute(0, 2, 3, 1),frameo3[:, :, 60:, :].permute(0, 2, 3, 1), frameo4.permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)

            psnr_list.extend(to_psnr(frame_out, clean))
            ssim_list.extend(to_ssim_skimage(frame_out, clean))
            if not os.path.exists('output/'):
               os.makedirs('output/')
            imwrite(frame_out, 'output/' + ''.join(name) + '.png', range=(0, 1))
avr_psnr = sum(psnr_list) / len(psnr_list)
avr_ssim = sum(ssim_list) / len(ssim_list)
print('PSNR: ', avr_psnr, 'SSIM: ', avr_ssim)    