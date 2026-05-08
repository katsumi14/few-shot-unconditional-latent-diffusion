import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import numpy as np
import shutil
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from DDPM_networks import NoiseScheduler, LDDPM_VAE, LDDPM_UNet
from LDDIM_test import LDDIM_generate
from mylib.visualizers import LossVisualizer
from mylib.data_io import autosaved_model_name, to_sigmoid_image, show_images
from mylib.utility import print_args, save_checkpoint, load_checkpoint


#The number of images in dataset 
IMG_NUM=14869
#The number of images used for traning in each epoch
TRAINING_IMG_NUM=15000

#The folder where the training results will be saved
# Please create this folder beforehand, as the code will throw an error if it doesn't exist
MODEL_DIR = './UNet_model'

# The path of pre-trained VAE model 
VAE_MODEL_PATH = './models/real_sketch_vae_model.pth'


# The size of the input image to the model (the height, width and channels of the image)
# Please change these values according to your environment and dataset
C = 3 # channels
H = 512 # height
W = 512 # width
LABEL_EMBED_DIM=64 # The dimension of the attribute information vector embedded in the model 

# The size of the latent feature map (the channels, height and width of the feature map)
ZC = 4 # channels
ZH = H // 8 # height
ZW = W // 8 # width

# The number of time steps in the diffusion process / reverse diffusion process
N_TIMESTEPS = 1000
N_GEN_TIMESTEPS = 50
# The dimension of the vector to encode the time step information
TIME_EMBED_DIM = 512

# The interval of epochs to automatically save the model during training when the auto-save mode is enabled
AUTO_SAVE_INTERVAL = 10

# The files used as checkpoints for saving the training state when the training process is interrupted and restarted
CHECKPOINT_EPOCH = os.path.join(MODEL_DIR, 'checkpoint_epoch.pkl')
CHECKPOINT_MODEL = os.path.join(MODEL_DIR, 'checkpoint_model.pth')
CHECKPOINT_OPT = os.path.join(MODEL_DIR, 'checkpoint_opt.pth')


def lr_cos_annealing(e:int, n_epochs:int, n_warmup_epochs:int=0, lr_max:float=1.0, lr_min:float=0.05):
    lr1 = lr_max if n_warmup_epochs <= 0 else lr_min + (lr_max - lr_min) * e / n_warmup_epochs
    lr2 = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * e / n_epochs))
    return min(lr1, lr2)


def main():

    # The device, the number of epochs, the batch size, etc. are obtained from the command line arguments and saved in variables.
    parser = argparse.ArgumentParser(description='Latent Denoising Diffusion Probabilistic Model Sample Code (UNet training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=400, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=32, type=int, help='minibatch size')
    parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'unet_model.pth'), type=str, help='file path of trained model')
    parser.add_argument('--autosave', '-s', help='this option makes the model automatically saved in each epoch', action='store_true')
    parser.add_argument('--restart', '-r', help='this option makes the training proccess restart from the most recent checkpoint', action='store_true')
    args = print_args(parser.parse_args())
    DEVICE = args['device']
    N_EPOCHS = args['epochs']
    BATCH_SIZE = args['batchsize']
    MODEL_PATH = args['model']
    AUTO_SAVE = args['autosave']
    RESTART_MODE = args['restart']

    INIT_EPOCH = 0 # The initial epoch number 
    LAST_EPOCH = INIT_EPOCH + N_EPOCHS # The final epoch number 

    # make the LDDPM-UNet model
    model = LDDPM_UNet(ZC=ZC, time_embed_dim=TIME_EMBED_DIM).to(DEVICE)

    # load the pre-trained VAE model 
    vae_model = LDDPM_VAE(C=C, ZC=ZC,label_embed_dim=LABEL_EMBED_DIM)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH))
    vae_model = vae_model.to(DEVICE)
    
    # specify the optimization algorithm (Adam is used here)
    optimizer = optim.AdamW(model.parameters(), lr=0.00002)

    # if -r option is specified, load the information from the previous checkpoint and restart the training
    if RESTART_MODE:
        INIT_EPOCH, LAST_EPOCH, model, optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_MODEL, CHECKPOINT_OPT, N_EPOCHS, model, optimizer)
        print('')
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, last_epoch=INIT_EPOCH-1, lr_lambda=lambda i: lr_cos_annealing(i, n_epochs=N_EPOCHS, n_warmup_epochs=0))
    # Loss function 
    loss_func = nn.MSELoss()

    # prepare the noise for the validation
    Z_valid = torch.randn((BATCH_SIZE, ZC, ZH, ZW)).to(DEVICE)
    # prepare the vissualizer for the loss function value graph
    visualizer = LossVisualizer(['train loss'], init_epoch=INIT_EPOCH, log_mode=True)

    # prepare the noise scheduler
    noise_scheduler = NoiseScheduler(device=DEVICE, method='linear', num_timesteps=N_TIMESTEPS)

    vae_model.eval()
    for epoch in range(INIT_EPOCH, LAST_EPOCH):

        print('Epoch {0}:'.format(epoch + 1))
        img_num=0# THe number of images used for training in the current epoch
        flag=0 # A flag to indicate whether the training process should be stopped when the number of images used for training reaches a certain number (15000 in this code)
        # Train
        model.train()
        sum_loss = 0
        # the number of iterations in current epoch must be changed according to the number of images used for training and the batch size
        for _ in tqdm(range(min(TRAINING_IMG_NUM, IMG_NUM)// BATCH_SIZE)):
            for param in model.parameters():
                param.grad = None
            batch_count=0 #The number of images used for training in the current mini-batch
            # prepare the input latent feature maps to train the U-Net efficiently
            mu=np.loadtxt(f'./Dataset/face_img_features/{img_num}.txt').reshape([4,64,64,]).astype(np.float32)
            z=mu
            img_num+=1
            batch_count+=1
            for i in range(1,BATCH_SIZE):
               if img_num>=min(TRAINING_IMG_NUM,IMG_NUM):
                   break
               comp_path=f'./Dataset/face_img_features/{img_num}.txt'
               img_num+=1
               batch_count+=1
               mu=np.loadtxt(comp_path).reshape([4,64,64]).astype(np.float64)
               z=np.append(z,mu)
            z_reshape=z.reshape((batch_count,4,64,64))
            Z=torch.from_numpy(z_reshape.astype(np.float32)).clone()
            X0 = Z.to(DEVICE)
            
            t = torch.randint(0, N_TIMESTEPS, (len(X0),), device=DEVICE).long() # setting the time step t randomly for each sample in the mini-batch
            noise = torch.randn_like(X0) # prepare the noise 
            Xt = noise_scheduler.get_noisy_sample(X0, t, noise) 
            noise_estimated = model(Xt, t) # predict the noise using U-Net
            # calculate the loss function and optimize the model parameters
            loss = loss_func(noise_estimated, noise) 
            loss.backward() 
            optimizer.step() 
            sum_loss += float(loss) * len(X0)
        lr_scheduler.step() # update the learning rate according to the learning rate scheduler
        avg_loss = sum_loss / img_num
        visualizer.add_value('train loss', avg_loss) # register the current value of the loss function with the visualizer
        print('train loss = {0:.6f}'.format(avg_loss))
        print('')

        # Validation
        model.eval()
        denoized_Z = LDDIM_generate(Z_valid, model, vae_model, noise_scheduler, n_timesteps=N_TIMESTEPS , n_gen_timesteps=N_GEN_TIMESTEPS)
        denoized_Z= denoized_Z.to(DEVICE)
        label=torch.zeros(BATCH_SIZE,dtype=torch.int64)
        label=label.to(DEVICE)
        Y=vae_model.decode(denoized_Z,label=label)
        Y_cpu = to_sigmoid_image(Y).to('cpu').detach()
        show_images(Y_cpu, num=len(Y_cpu), num_per_row=8, title='unet_epoch_{}'.format(epoch+1), save_only=True, save_dir=MODEL_DIR)
        del Y
        torch.cuda.empty_cache()
        
        # display the training progress by showing the loss function value graph
        visualizer.show(sec=1) # 1秒間停止

        # save the loss function history and visualization results to files
        visualizer.save(v_file=os.path.join(MODEL_DIR, 'unet_loss_graph.png'), h_file=os.path.join(MODEL_DIR, 'unet_loss_history.csv'))

        # save the model parameters and the training state to files as a checkpoint at the end of each epoch
        save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_MODEL, CHECKPOINT_OPT, epoch+1, model, optimizer)

        # save the model during training automotically at the specified interval if the -s option is specified
        if AUTO_SAVE and (epoch+1) % AUTO_SAVE_INTERVAL == 0:
            shutil.copy(CHECKPOINT_MODEL, autosaved_model_name(MODEL_PATH, epoch + 1))

    # save the trained neural network model to a file at the end of the training
    torch.save(model.to('cpu').state_dict(), MODEL_PATH)
    

if __name__ == '__main__':
    main()
