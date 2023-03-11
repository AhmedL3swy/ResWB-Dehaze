from env import *
from Models.sobel import *

#data augmentation for image rotate
def custom_augment(hazy, clean, clean_gray):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        clean_gray = transforms.functional.rotate(clean_gray, rotate_degree)
        return hazy, clean, clean_gray
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        clean_gray = vertical_flip(clean_gray)
        return hazy, clean, clean_gray
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        clean_gray = horizontal_flip(clean_gray)
        return hazy, clean, clean_gray
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean, clean_gray

class custom_dehaze_train_dataset(Dataset):
    def __init__(self, HAZY_path = None, GT_path = None, Image_Size = (256,256), is_train = True):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.HAZY_path = Path(HAZY_path)
        self.GT_path = Path(GT_path)
        self.HAZY_Image = sorted(self.HAZY_path.glob("*.png")) # list all the files present in HAZY images folder...
        self.GT_Image = sorted(self.GT_path.glob("*.png")) # list all the files present in GT images folder...
        self.Image_Size = Image_Size
        self.is_train = is_train

    def __getitem__(self, index):
        hazy = Image.open(self.HAZY_Image[index])
        clean = Image.open(self.GT_Image[index])
        if self.is_train:
            # clean_gray = clean.convert('L')
            clean_gray = clean
            #crop a patch
            i,j,h,w = transforms.RandomCrop.get_params(hazy, output_size = self.Image_Size)
            hazy_ = TF.crop(hazy, i, j, h, w)
            clean_ = TF.crop(clean, i, j, h, w)
            clean_gray_ = TF.crop(clean_gray, i, j, h, w)

            #data argumentation
            hazy_arg, clean_arg, clean_gray_arg = custom_augment(hazy_, clean_, clean_gray_)
            hazy = self.transform(hazy_arg)
            clean = self.transform(clean_arg)
            rgb_edged_cv2_x = cv2.Sobel(np.float32(clean_gray_arg), cv2.CV_64F, 1, 0, ksize=3)
            rgb_edged_cv2_y = cv2.Sobel(np.float32(clean_gray_arg), cv2.CV_64F, 0, 1, ksize=3)
            rgb_edged_cv2 = np.sqrt(np.square(rgb_edged_cv2_x), np.square(rgb_edged_cv2_y))
            clean_gray = self.transform(rgb_edged_cv2)
            return hazy,clean,clean_gray/255
        else:
          hazy = self.transform(hazy)
          clean = self.transform(clean)
          return hazy,clean

    def __len__(self):
        return len(self.HAZY_Image) # return length of dataset

class dehaze_test_dataset(Dataset):
    def __init__(self, HAZY_PATH = None, GT_PATH = None):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.root_hazy = Path(HAZY_PATH)
        self.root_GT = Path(GT_PATH)
        self.list_test = sorted(self.root_hazy.glob("*.png")) # list all the files present in HAZY images folder...
        self.list_GT = sorted(self.root_GT.glob("*.png")) # list all the files present in GT images folder...
        self.file_len = len(self.list_test)
    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.list_test[index])
        hazy = self.transform(hazy)
        #----------- Gives cuda out of memory -----------
        hazy_up=hazy[:,0:1152,:]
        hazy_down=hazy[:,48:1200,:]

        # ----------- Doesn't give cuda out of memory but the separating line is visible -----------
        # hazy_up=hazy[:,0:768,:]
        # hazy_down=hazy[:,432:1200,:]
        name=self.list_test[index].stem
        if len(self.list_GT) == 0:
          return hazy_up,hazy_down,name
        else:
          clean=Image.open(self.list_GT[index])
          clean = self.transform(clean)
          return hazy_up, hazy_down, name, clean 
    def __len__(self):
        return self.file_len

class CustomDataLoader(Dataset):
    def __init__(self, HAZY_path = None, GT_path = None, image_size = None, resize = None):
        self.HAZY_path = Path(HAZY_path)
        self.GT_path = Path(GT_path)
        self.HAZY_Image = sorted(self.HAZY_path.glob("*.png")) # list all the files present in HAZY images folder...
        self.GT_Image = sorted(self.GT_path.glob("*.png")) # list all the files present in GT images folder...
        assert len(self.HAZY_Image) == len(self.GT_Image)  
        self.resize = resize
        if(self.resize):
            self.data_transforms = transforms.Compose([transforms.Resize(image_size),
                                                        transforms.ToTensor()])
        else:
            self.data_transforms = transforms.Compose([transforms.ToTensor()])

    def load_image(self, index: int, image_type = "HAZY") -> Image.Image:
        "Opens an image via a path and returns it."

        if image_type == "HAZY":
          image_path = self.HAZY_Image[index]
        elif image_type == "GT":
          image_path = self.GT_Image[index]

        return Image.open(image_path)
        
    def __len__(self):
        return len(self.HAZY_Image) # return length of dataset
    
    def __getitem__(self, index):
        HAZY = Image.open(self.HAZY_Image[index])
        GT = Image.open(self.GT_Image[index]) 
        return self.data_transforms(HAZY), self.data_transforms(GT), self.HAZY_Image[index].stem