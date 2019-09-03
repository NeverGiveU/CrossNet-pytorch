# -*- coding:utf-8 -*-


"""

@author: Jan
@file: data.py
@time: 2019/9/1 13:22
"""
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import torch
import numpy as np


#### 01. Create a dataset
## BaseDataset: we define some common functions here
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return "BaseDataset"

    def initialize(self, opt, option=''):
        pass


## SelfDataset
class UnAlignedDataset(BaseDataset):
    def name(self):
        return "UnAlignedDataset"

    def initialize(self, opt):
        self.opt = opt

        ## get dir
        self.dataroot = opt.dataroot

        ## get images
        if opt.mode == 'train':
            dir_A = os.path.join(opt.dataroot, opt.trainA)
            dir_B = os.path.join(opt.dataroot, opt.trainB)
        elif opt.mode == 'test':
            dir_A = os.path.join(opt.dataroot, opt.testA)
            dir_B = os.path.join(opt.dataroot, opt.testB)

        A_paths = os.listdir(dir_A)
        B_paths = os.listdir(dir_B)
        self.length = min(len(A_paths), len(B_paths))

        ## get full path
        for i in range(len(A_paths)):
            A_paths[i] = os.path.join(dir_A, A_paths[i])
        for i in range(len(B_paths)):
            B_paths[i] = os.path.join(dir_B, B_paths[i])
        self.A_paths = A_paths
        self.B_paths = B_paths

        self.input_nc = self.opt.input_nc

        ## define transform
        transforms_list = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        A_pth = self.A_paths[index % self.length]
        B_pth = self.B_paths[index % self.length]

        x_img = Image.open(A_pth).convert('RGB')
        x_img = x_img.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        x = self.transform(x_img)

        y_img = Image.open(B_pth).convert('RGB')
        y_img = y_img.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        y = self.transform(y_img)

        ## crop
        h, w = x.size(1), x.size(2)
        h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))
        w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
        x = x[:, h_offset:h_offset + self.opt.crop_size, w_offset:w_offset + self.opt.crop_size]
        y = y[:, h_offset:h_offset + self.opt.crop_size, w_offset:w_offset + self.opt.crop_size]

        ## expand to 4-dim tensor
        if self.opt.input_nc == 1:
            # RGB to gray
            tmp_x = x[0, ...] * 0.299 + x[1, ...] * 0.587 + x[2, ...] * 0.114
            x = tmp_x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            tmp_y = y[0, ...] * 0.299 + y[1, ...] * 0.587 + y[2, ...] * 0.114
            x = tmp_y.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        return {'A': x, 'B':y, 'A_pth': A_pth, 'B_pth': B_pth}


#### 0.2 Create a Dataloader
## BaseDataLoader
class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt

    def load_data(self):
        return None


## Dataloader for self data
class UnAlignedDataLoader(BaseDataLoader):
    def name(self):
        return "UnAlignedDataLoader"

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)  # get the copy of opt->self.opt
        # add dataset and nitialize it
        self.dataset = UnAlignedDataset()
        self.dataset.initialize(opt)
        # define a data loader
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.n_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, d in enumerate(self.dataloader):
            yield d  # data


#### 0.3 Image pool
class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size):
        """
        Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.count = 0
            self.images = []

    def query(self, images):
        """
        Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.
        By p = 50/100, the buffer will return input images.
        By p = 50/100, the buffer will return images previously stored in the buffer, and insert the current images to the buffer.
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.count < self.pool_size: #-> no full, some space left
                self.count = self.count + 1
                self.images.append(image)   #-> add new image into history
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: #-> 50% chance
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone() #-> replace
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image) #-> by another 50% chance, no change

        return_images = torch.cat(return_images, dim=0)
        return return_images


#### Test data loader
"""
from config import parser
opt = parser.parse_args()
data_loader = UnAlignedDataLoader()
data_loader.initialize(opt)

data_set = data_loader.load_data()

for i, data in enumerate(data_set):
    print(i, data['A_pth'], data['B_pth'])
    if i >= 2:
        break
"""

#### visualize functions
def tensor2image_RGB(tensor):
    '''
    tensor torch.tensor(C, H, W)
    '''
    arr = np.array(tensor).transpose(1, 2, 0)
    arr = (arr + 1) / 2
    return arr
