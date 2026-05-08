import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from DDPM_networks import LDDPM_VAE
from mylib.data_io import CSVBasedDataset, autosaved_model_name, to_sigmoid_image, show_images
from mylib.utility import print_args



# The CSV file paths that contain the dataset information (image file paths and labels)
DATASET_CSV = './Dataset/crop_sketches_10.csv'
# the directory path where the image files are located
DATA_DIR = './Dataset/'
# The file path of pre-trained VAE model
MODEL_PATH = './models/real_face_vae_model.pth'
# The file path of saving the feature maps that is enoded from images by VAE
SAVE_DIR ='./Dataset/sketch_features_10/'
# the size of images in the dataset
C = 3 # channel
H = 512 # height
W = 512 # width
#the dimension of the attribute information vector embedded in the model
LABEL_EMBED_DIM=64
# the channel of feature maps that VAE enoder 
ZC = 4

def main():
    
    # get the setting infomation of DEVICE and batch size from comand line argument  and save it to variable
    parser = argparse.ArgumentParser(description='Latent Denoising Diffusion Probabilistic Model Sample Code (VAE training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=8, type=int, help='minibatch size')
    args = print_args(parser.parse_args())
    DEVICE = args['gpu']
    BATCH_SIZE = args['batchsize']

    # load the CSV file and prepare the tarin dataset 
    dataset = CSVBasedDataset(
        filename = DATASET_CSV,
        items = [
            'Original DataPath' # X
        ],
        dtypes = [
            'image' # type of X
        ],
        dirname = DATA_DIR,
        img_transform=transforms.CenterCrop((H, W)), 
        img_range=[-1, 1],
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # load the pre trained VAE model
    model = LDDPM_VAE(C=C, ZC=ZC,label_embed_dim=LABEL_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    

    
    # detect features in the dataset
    model.eval()
    img_num=0
    with torch.inference_mode():
         for X in tqdm(dataloader):
             X=X.to(DEVICE)
             mu = model.encode(X,testmode=True)
             mu_copy=mu.to('cpu').detach().numpy().copy()
             mu=mu.to(DEVICE)
             for i in range(len(mu_copy)):
                 comp_path= os.path.join(SAVE_DIR, f'{img_num}.txt')
                 np.savetxt(comp_path,mu_copy[i].reshape([4,4096]))
                 img_num +=1
         

if __name__ == '__main__':
    main()
