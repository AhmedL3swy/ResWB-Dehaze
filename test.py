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
import yaml
with open("testcfg.yaml","r") as f:
    config=yaml.safe_load(f)

VAL_HAZY_IMAGES_PATH = config["VAL_HAZY_IMAGES_PATH"]
VAL_GT_IMAGES_PATH = config["VAL_GT_IMAGES_PATH"]
RESIZE = config["RESIZE"] 

TEST_IMAGE_SIZE = config["TEST_IMAGE_SIZE"] 
VAL_BATCH_SIZE = config["VAL_BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]

# --- output picture and check point --- #
G_model_save_dir = config["G_model_save_dir"]
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Custom_fusion_net().float()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

# --- Load testing data --- #
val_data = CustomDataLoader(HAZY_path = VAL_HAZY_IMAGES_PATH,
                            GT_path = VAL_GT_IMAGES_PATH,
                            image_size = TEST_IMAGE_SIZE,
                            resize = RESIZE)

val_loader = DataLoader(val_data, 
                        batch_size = VAL_BATCH_SIZE, 
                        num_workers = NUM_WORKERS)
MyEnsembleNet = MyEnsembleNet.to(device)

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
    for batch_idx, (hazy, clean, data_name) in enumerate(val_loader): 
        clean = clean.to(device)
        hazy = hazy.to(device)
        frame_out, _ = MyEnsembleNet(hazy)
        psnr_list.extend(to_psnr(frame_out, clean))
        ssim_list.extend(to_ssim_skimage(frame_out, clean))
        if not os.path.exists('test/'):
            os.makedirs('test/')
        imwrite(frame_out, 'test/' + ''.join(data_name) + '.png', range=(0, 1))

avr_psnr = sum(psnr_list) / len(psnr_list)
avr_ssim = sum(ssim_list) / len(ssim_list)
print('PSNR: ', avr_psnr, 'SSIM: ', avr_ssim) 