from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from scipy.io import loadmat
#Dataset created using .mat file created using dataset_create.py
class SVHNDataset(Dataset):
   
    def __init__(self, mat_file,transform=None):
       self.data={}
       self.data['X'],self.data['y'] = loadmat(mat_file)['X'],loadmat(mat_file)['y']
       
       self.transform = transform
      
    def __len__(self):
        return self.data['X'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw_img = np.transpose(self.data['X'][idx,:,:,:],(2,0,1))
        normalized_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
        normalized_img = normalized_img.astype('float')
        if self.transform:
            normalized_img = self.transform(torch.from_numpy(normalized_img))
        return normalized_img,self.data['y'][idx]
