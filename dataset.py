import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from utils import ToHSV, ToComplex
import os

def fft_real_imag_analysis(image_tensor):
    img = image_tensor  # [C, H, W]
    fft_img = torch.fft.rfft2(img, dim=(-2, -1), norm='ortho')

    real_part = torch.real(fft_img)
    imag_part = torch.imag(fft_img)

    fft_real = torch.complex(real_part, torch.zeros_like(imag_part))
    fft_imag = torch.complex(torch.zeros_like(real_part), imag_part)

    recon_real = torch.fft.irfft2(fft_real, s=img.shape[-2:], dim=(-2, -1), norm='ortho')
    recon_imag = torch.fft.irfft2(fft_imag, s=img.shape[-2:], dim=(-2, -1), norm='ortho')

    # Extract top half (upright image)
    H = recon_real.shape[-2]
    top_half_real = recon_real[:, :, :]
    top_half_imag = recon_imag[:, :, :]

    return top_half_real + 1j * top_half_imag  # Return as a complex tensor

    

class CamVidDataset(Dataset):
    def __init__(self, dir, transforms=None,
                 class_dict='CamVid/class_dict.csv', 
                 mode='train'):
        super().__init__()
        self.image_dir = f'{dir}/{mode}'
        self.image_paths = natsorted(glob.glob(self.image_dir+'/*.png'))
        self.mask_dir = f'{dir}/{mode}_labels'
        self.mask_paths = natsorted(glob.glob(self.mask_dir+'/*.png'))
        self.color_map, self.color_to_idx = self.load_colormap(class_dict)
        if transforms is not None:
            self.img_transforms = transforms[0]
            self.mask_transforms = transforms[1]
        else:
            self.img_transforms = None
            self.mask_transforms = None

    def load_colormap(self, csv_path):
        df = pd.read_csv(csv_path)
        color_map = [tuple(row) for row in df[['r', 'g', 'b']].values]
        color_to_idx = {color:idx for idx, color in enumerate(color_map)}
        return color_map, color_to_idx
    
    def encode_segmap(self, mask):
        mask = np.array(mask)
        mask_encoded = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, idx in self.color_to_idx.items():
            mask_encoded[np.all(mask==color, axis=-1)]=idx

        return torch.tensor(mask_encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_transforms is not None:
            image = self.img_transforms(image)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        mask = self.encode_segmap(mask)
        return image, mask, img_path
    

    

if __name__=='__main__':
    # get RGB images first
    os.makedirs('Dataset/CamVid_RGB/train', exist_ok=True)
    os.makedirs('Dataset/CamVid_RGB/test', exist_ok=True)

    transforms_img = T.Compose([
        T.ToTensor(),
        T.Resize((256,256))
    ])
    transforms_mask = T.Compose([
        T.Resize((256, 256))
    ])

    dataset = CamVidDataset(
        dir = 'CamVid', mode='train', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid
    # print(len(dataset))
    # image = dataset[0][0]
    # mask = dataset[0][1]
    # print(image.shape, mask.shape)

    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/CamVid_RGB/train/{save_path}.pth'
        data = {}
        data['mask'] = mask
        data['img'] = image
        # data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)
        # break

    dataset = CamVidDataset(
        dir = 'CamVid', mode='test', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid
    # print(len(dataset))
    # image = dataset[0][0]
    # mask = dataset[0][1]
    # print(image.shape, mask.shape)

    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/CamVid_RGB/test/{save_path}.pth'
        data = {}
        data['mask'] = mask
        data['img'] = image
        # data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)


    os.makedirs('Dataset/Complex_CamVid_iHSV/train', exist_ok=True)
    os.makedirs('Dataset/Complex_CamVid_iHSV/test', exist_ok=True)
    transforms_img = T.Compose([
        T.ToTensor(),
        T.Resize((256,256)),
        ToHSV(),
        ToComplex()
    ])
    transforms_mask = T.Compose([
        T.Resize((256, 256))
    ])

    dataset = CamVidDataset(
        dir = 'CamVid', mode='train', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid


    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/Complex_CamVid_iHSV/train/{save_path}.pth'
        data = {}
        data['mask'] = mask
        data['img'] = image
        # data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)
        # break

    dataset = CamVidDataset(
        dir = 'CamVid', mode='test', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid
    # print(len(dataset))
    # image = dataset[0][0]
    # mask = dataset[0][1]
    # print(image.shape, mask.shape)

    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/Complex_CamVid_iHSV/test/{save_path}.pth'
        data = {}
        data['mask'] = mask
        data['img'] = image
        # data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)


    os.makedirs('Dataset/Complex_CamVid_inv_FFT/train', exist_ok=True)
    os.makedirs('Dataset/Complex_CamVid_inv_FFT/test', exist_ok=True)
    transforms_img = T.Compose([
        T.ToTensor(),
        T.Resize((256,256))
    ])
    transforms_mask = T.Compose([
        T.Resize((256, 256))
    ])

    dataset = CamVidDataset(
        dir = 'CamVid', mode='train', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid


    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/Complex_CamVid_inv_FFT/train/{save_path}.pth'
        data = {}
        data['mask'] = mask
        # data['img'] = image
        data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)
        # break

    dataset = CamVidDataset(
        dir = 'CamVid', mode='test', 
        transforms=(transforms_img, transforms_mask)
    ) # assuming the train and test files are inside a folder called Camvid
    # print(len(dataset))
    # image = dataset[0][0]
    # mask = dataset[0][1]
    # print(image.shape, mask.shape)

    for image, mask, path in tqdm(dataset):
        save_path = path.split('/')[-1].split('.')[0]
        save_path = f'Dataset/Complex_CamVid_inv_FFT/test/{save_path}.pth'
        data = {}
        data['mask'] = mask
        # data['img'] = image
        data['img'] = fft_real_imag_analysis(image)
        # print(fft_real_imag_analysis(image).shape)
        torch.save(data, save_path)