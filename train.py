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
# --- train --- #
import yaml
with open("traincfg.yaml","r") as f:
    config=yaml.safe_load(f)
train_epoch = config["train_epoch"] # Currently at 1700 epochs and should reach 1800 after this
best_psnr = config["best_psnr"]
# TRAIN_HAZY_IMAGES_PATH = config["TRAIN_HAZY_IMAGES_PATH"]
# TRAIN_GT_IMAGES_PATH = config["TRAIN_GT_IMAGES_PATH"]
# VAL_HAZY_IMAGES_PATH = config["VAL_HAZY_IMAGES_PATH"]
# VAL_GT_IMAGES_PATH = config["VAL_GT_IMAGES_PATH"]
TRAIN_HAZY_IMAGES_PATH = config["TRAIN_HAZY_IMAGES_PATH"]
TRAIN_GT_IMAGES_PATH = config["TRAIN_GT_IMAGES_PATH"]
VAL_HAZY_IMAGES_PATH = config["VAL_HAZY_IMAGES_PATH"]
VAL_GT_IMAGES_PATH = config["VAL_GT_IMAGES_PATH"]
IMAGE_SIZE = (config["IMAGE_SIZE"],config["IMAGE_SIZE"])
TRAIN_BATCH_SIZE = config["TRAIN_BATCH_SIZE"]
VAL_BATCH_SIZE = config["VAL_BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]
SHUFFLE = config["SHUFFLE"]
# --- output picture and check point --- #
G_model_save_dir = config["G_model_save_dir"]
D_model_save_dir = config["D_model_save_dir"]
G_best_model_save_dir = config["G_best_model_save_dir"]
D_best_model_save_dir = config["D_best_model_save_dir"]
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Custom_fusion_net().float()
DNet = Discriminator()

print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)

# --- Load training data --- #
dataset = custom_dehaze_train_dataset(HAZY_path = TRAIN_HAZY_IMAGES_PATH, GT_path = TRAIN_GT_IMAGES_PATH, Image_Size = IMAGE_SIZE,is_train = True)
train_loader = DataLoader(dataset=dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


# --- Load testing data --- #
val_data = dehaze_test_dataset(VAL_HAZY_IMAGES_PATH, VAL_GT_IMAGES_PATH)
val_loader = DataLoader(dataset=val_data, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)
# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(G_model_save_dir))
    DNet.load_state_dict(torch.load(D_model_save_dir))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Define the perceptual loss network --- #
backbone_model = vgg16(weights = VGG16_Weights.DEFAULT)

backbone_model = backbone_model.features[:16].to(device)
for param in backbone_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(backbone_model)
loss_network.eval()
msssim_loss = msssim

# --- Start training --- #
for epoch in range(train_epoch):
    psnr_list = []
    ssim_list = []
    MyEnsembleNet.train()
    DNet.train()
    avg_loss = 0
    print("We are in epoch: " + str(epoch+1))

    for batch_idx, (hazy, clean, clean_sobel) in enumerate(train_loader): 
            hazy = hazy.to(device)
            clean = clean.to(device)
            clean_sobel = clean_sobel.to(device)
            output, hazy_sobel = MyEnsembleNet(hazy.float())
            real_out = DNet(clean)
            fake_out = DNet(output)
            real_loss = F.binary_cross_entropy(real_out, torch.ones(real_out.size()).to(device))
            fake_loss = F.binary_cross_entropy(fake_out, torch.zeros(fake_out.size()).to(device))
            D_loss = (real_loss + fake_loss) / 2
            DNet.zero_grad()
            D_loss.backward(retain_graph=True)
            smooth_loss_l1 = F.smooth_l1_loss(output, clean)
            perceptual_loss = loss_network(output, clean)
            msssim_loss_ = 1 - msssim_loss(output, clean, normalize=True)
            calc_psnr = to_psnr(output, clean)
            calc_ssim = to_ssim_skimage(output, clean)
            sobel_l1_loss = F.smooth_l1_loss(hazy_sobel, clean_sobel.float())
            sobel_msssim_loss = 1 - msssim_loss(hazy_sobel, clean_sobel.float(), normalize=True)
            total_loss = (smooth_loss_l1 + sobel_l1_loss)/2 + 0.05 * perceptual_loss +  (msssim_loss_ + sobel_msssim_loss)/2
            avg_loss += total_loss.item()
            MyEnsembleNet.zero_grad()
            total_loss.backward()
            G_optimizer.step()
            D_optim.step()
            psnr_list.extend(calc_psnr)
            ssim_list.extend(calc_ssim)
    
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    print('AVG PSNR: ', avr_psnr, 'AVG SSIM: ', avr_ssim, 'AVG Loss: ', avg_loss/len(psnr_list))

    if (epoch+1) % 5 == 0: 
      print("-----Testing-----")     
      with torch.inference_mode():
          psnr_list = []
          ssim_list = []
          MyEnsembleNet.eval()
          for batch_idx, (hazy_up,hazy_down,name,clean) in enumerate(val_loader):
              hazy_up = hazy_up.to(device)
              hazy_down = hazy_down.to(device)
              clean = clean.to(device)
              frame_out_up = MyEnsembleNet(hazy_up)
              frame_out_down = MyEnsembleNet(hazy_down)
              frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)
              psnr_list.extend(to_psnr(frame_out, clean))
              ssim_list.extend(to_ssim_skimage(frame_out, clean))
              imwrite(frame_out, '/content/drive/MyDrive/Graduation Project/CANT_Haze/Restormer Twice & AWB Results/' + ''.join(name) + '.png', range=(0, 1))

      avr_psnr = sum(psnr_list) / len(psnr_list)
      avr_ssim = sum(ssim_list) / len(ssim_list)
      print('PSNR: ', avr_psnr, 'SSIM: ', avr_ssim)    
      torch.save(MyEnsembleNet.state_dict(), G_model_save_dir)
      torch.save(DNet.state_dict(), D_model_save_dir)
      print("-----Model Saved-----")
      if(avr_psnr > best_psnr):
          best_psnr = avr_psnr
          torch.save(MyEnsembleNet.state_dict(), G_best_model_save_dir)
          torch.save(DNet.state_dict(), D_best_model_save_dir)
          print("-----Best Model Saved-----")