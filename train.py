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
train_epoch = 100 # Currently at 1700 epochs and should reach 1800 after this
best_psnr = 20.75
# TRAIN_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/train_dense/haze/"
# TRAIN_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/train_dense/GT/"
# VAL_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/valid_dense/HAZY/"
# VAL_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/valid_dense/GT/"
TRAIN_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/train_NH/haze/"
TRAIN_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/train_NH/clear_images/"
VAL_HAZY_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/Test_Hazy/"
VAL_GT_IMAGES_PATH = "/content/drive/MyDrive/Graduation Project/data/NH-HAZE/Test_GT/"
IMAGE_SIZE = (256,256)
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_WORKERS = 0
SHUFFLE = True
# --- output picture and check point --- #
G_model_save_dir = "/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/Generator_NH_Restormer_Twice_HM_GBlock_AWB_Sobel.pth"
D_model_save_dir = "/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/Discriminator_NH_Restormer_Twice_HM_GBlock_AWB_Sobel.pth"
G_best_model_save_dir = "/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/Generator_NH_Restormer_Twice_HM_GBlock_AWB_Sobel_Best.pth"
D_best_model_save_dir = "/content/drive/MyDrive/Graduation Project/CANT_Haze/Weights/Discriminator_NH_Restormer_Twice_HM_GBlock_AWB_Sobel_Best.pth"
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Custom_fusion_net().float()
DNet = Discriminator()
# for name, param in MyEnsembleNet.named_parameters():
#     if param.requires_grad and 'haze_density' in name:
#         param.requires_grad = False
# non_frozen_parameters = [p for p in MyEnsembleNet.parameters() if p.requires_grad]
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
# print('Nonfrozen parameters:', sum(param.numel() for param in non_frozen_parameters))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
# G_optimizer = torch.optim.Adam(non_frozen_parameters, lr=0.0001)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
# scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 6000], gamma=0.5)

dataset = custom_dehaze_train_dataset(HAZY_path = TRAIN_HAZY_IMAGES_PATH, GT_path = TRAIN_GT_IMAGES_PATH, Image_Size = IMAGE_SIZE,is_train = True)
train_loader = DataLoader(dataset=dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# train_data = CustomDataLoader(HAZY_path = TRAIN_HAZY_IMAGES_PATH,
#                               GT_path = TRAIN_GT_IMAGES_PATH,
#                               image_size = IMAGE_SIZE,
#                               white_balance = False,
#                               crop = True)

# train_loader = DataLoader(train_data, 
#                           batch_size = TRAIN_BATCH_SIZE, 
#                           num_workers = NUM_WORKERS,
#                           shuffle = SHUFFLE)

# --- Load testing data --- #
val_data = CustomDataLoader(HAZY_path = VAL_HAZY_IMAGES_PATH,
                            GT_path = VAL_GT_IMAGES_PATH,
                            image_size = (768,1024),
                            white_balance = False,
                            crop = False,
                            resize = True)

val_loader = DataLoader(val_data, 
                        batch_size = VAL_BATCH_SIZE, 
                        num_workers = NUM_WORKERS)

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
# backbone_model = maxvit_t2(weights = models.MaxVit_T_Weights).to(device)
for param in backbone_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(backbone_model)
# loss_network = CustomLossNetwork(backbone_model)
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
    # for batch_idx, (hazy, clean, data_name) in enumerate(train_loader):
    for batch_idx, (hazy, clean, clean_sobel) in enumerate(train_loader): 
        # for i in range(len(hazy)): 
            hazy = hazy.to(device)
            clean = clean.to(device)
            clean_sobel = clean_sobel.to(device)
            # (hazy,clean) = lazy(hazy, clean, batch=batch_idx)
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
            # print('PSNR: ', calc_psnr, 'SSIM: ', calc_ssim, 'MS_SSIM_Loss: ', msssim_loss_.item(), 'smooth_loss_l1: ', smooth_loss_l1.item(), 'perceptual_loss: ', perceptual_loss.item(), 'total_loss', total_loss.item())
            psnr_list.extend(calc_psnr)
            ssim_list.extend(calc_ssim)
    
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    print('AVG PSNR: ', avr_psnr, 'AVG SSIM: ', avr_ssim, 'AVG Loss: ', avg_loss/len(psnr_list))
    # scheduler_G.step()
    if (epoch+1) % 5 == 0: 
            print("-----Testing-----")     
            with torch.inference_mode():
                psnr_list = []
                ssim_list = []
                MyEnsembleNet.eval()
                for batch_idx, (hazy, clean, data_name) in enumerate(val_loader): 
                    clean = clean.to(device)
                    hazy = hazy.to(device)
                    frame_out, _ = MyEnsembleNet(hazy)
                    # if not os.path.exists('test/'):
                    #     os.makedirs('test/')
                    imwrite(frame_out, '/content/drive/MyDrive/Graduation Project/CANT_Haze/Restormer Twice & AWB Results/' + ''.join(data_name) + '.png', range=(0, 1))
                    psnr_list.extend(to_psnr(frame_out, clean))
                    ssim_list.extend(to_ssim_skimage(frame_out, clean))
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