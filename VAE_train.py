import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from DDPM_networks import LDDPM_VAE
from mylib.loss_functions import VAELoss
from mylib.visualizers import LossVisualizer
from mylib.data_io import CSVBasedDataset, autosaved_model_name, to_sigmoid_image, show_images
from mylib.utility import print_args, save_datasets, save_sub_datasets, load_datasets_from_file,load_sub_datasets_from_file, save_checkpoint, load_checkpoint


# The CSV file paths that contain the dataset information (image file paths and labels)
# Please change these paths according to your environment and dataset
SOURCE_DATASET_CSV = './Dataset/crop_images.csv'
TARGET_DATASET_CSV = './Dataset/crop_sketches.csv'
# The prefix string to be added to the image file names (the path to the directory where the image files are located)
# Please change this path according to your environment and dataset
DATA_DIR = './Dataset/'
LABEL_EMBED_DIM=64 #the dimension of the attribute information vector embedded in the model
# The folder where the training results will be saved
# Please create this folder beforehand, as the code will throw an error if it doesn't exist
MODEL_DIR = './VAE_model'

# The size of the input image to the model (the height, width and channels of the image)
# Please change these values according to your environment and dataset
C = 3 # The number of channels
H = 512 # The height of the image
W = 512 # The width of the image
N = 2 # The number of domains (source and target)
# The number of channels in the latent feature map
ZC = 4
# the set of mixed precision training
USE_AMP = True
FLOAT_DTYPE = torch.bfloat16 
LOSS_SCALER = torch.amp.grad_scaler.GradScaler(enabled=USE_AMP, device='cuda', init_scale=2**12)
ADAM_EPS = 1e-4 if USE_AMP and (FLOAT_DTYPE == torch.float16) else 1e-8

# The interval at which the neural network model will be automatically saved
AUTO_SAVE_INTERVAL = 10

ATTRIBUTES=['Real_face','Sketch_face']
# The files to restart the training of the model from the most recent checkpoint in case of interruption
CHECKPOINT_EPOCH = os.path.join(MODEL_DIR, 'checkpoint_epoch.pkl')
CHECKPOINT_MODEL = os.path.join(MODEL_DIR, 'checkpoint_model.pth')
CHECKPOINT_OPT = os.path.join(MODEL_DIR, 'checkpoint_opt.pth')


def main():
    
    # Get the device, number of epochs, batch size, etc. from command line arguments and save them in variables
    parser = argparse.ArgumentParser(description='Latent Denoising Diffusion Probabilistic Model Sample Code (VAE training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=16, type=int, help='minibatch size')
    parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'vae_model.pth'), type=str, help='file path of trained model')
    parser.add_argument('--autosave', '-s', help='this option makes the model automatically saved in each epoch', action='store_true')
    parser.add_argument('--restart', '-r', help='this option makes the training proccess restart from the most recent checkpoint', action='store_true')
    args = print_args(parser.parse_args())
    DEVICE = args['device']
    N_EPOCHS = args['epochs']
    BATCH_SIZE = args['batchsize']
    MODEL_PATH = args['model']
    AUTO_SAVE = args['autosave']
    RESTART_MODE = args['restart']

    INIT_EPOCH = 0 # The initial epoch number (if not restarting, it starts from 0)
    LAST_EPOCH = INIT_EPOCH + N_EPOCHS # The final epoch number
    if RESTART_MODE:
        source_train_dataset, source_valid_dataset = load_datasets_from_file(MODEL_DIR)
        target_train_dataset, target_valid_dataset = load_sub_datasets_from_file(MODEL_DIR)
        source_train_size=len(source_train_dataset)
        source_valid_size=len(source_valid_dataset)
    else:
        # prepare the training dataset by reading the CSV file (source domain)
        source_dataset = CSVBasedDataset(
            filename = SOURCE_DATASET_CSV,
            items = [
                'Original DataPath' # X
            ],
            dtypes = [
                'image' # The type of X
            ],
            dirname = DATA_DIR,
            img_transform=transforms.CenterCrop((H, W)), # use only the central (H, W) pixels of the image
            img_range=[-1, 1],
        )
        # The training dataset is split, and one part is used for validation
        source_dataset_size = len(source_dataset)
        source_valid_size = int(0.02 * source_dataset_size) # use 2% of the dataset for validation
        source_train_size = source_dataset_size - source_valid_size # use the remaining 98% for training
        source_train_dataset, source_valid_dataset = random_split(source_dataset, [source_train_size, source_valid_size])

        # prepare the training dataset by reading the CSV file (target domain)
        target_dataset = CSVBasedDataset(
            filename = TARGET_DATASET_CSV,
            items = [
                'Original DataPath' # X
            ],
            dtypes = [
                'image' # The type of X
            ],
            dirname = DATA_DIR,
            img_transform=transforms.CenterCrop((H, W)), # use only the central (H, W) pixels of the image
            img_range=[-1, 1],
        )
        # The training dataset is split, and one part is used for validation
        target_dataset_size = len(target_dataset)
        target_valid_size = int(0.2 * target_dataset_size) # use 20% of the dataset for validation
        target_train_size = target_dataset_size - target_valid_size # use the remaining 80% for training
        target_train_dataset, target_valid_dataset = random_split(target_dataset, [target_train_size, target_valid_size])
        # save the source and target datasets to files
        save_datasets(MODEL_DIR, source_train_dataset, source_valid_dataset)
        save_sub_datasets(MODEL_DIR, target_train_dataset, target_valid_dataset)
    # prepare data loaders to use the training and validation datasets in mini-batches
    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE//2, shuffle=True, pin_memory=True)
    source_valid_dataloader = DataLoader(source_valid_dataset, batch_size=BATCH_SIZE//2, shuffle=False, pin_memory=True)
    # The process of data augmentation 
    image_transform = transforms.RandomHorizontalFlip(p=0.5) # with a probability of 0.5, flip the image horizontally (left-right)

    # create the model and move it to the specified device (GPU or CPU)
    model = LDDPM_VAE(C=C, ZC=ZC,label_embed_dim=LABEL_EMBED_DIM).to(DEVICE)
    # create the optimizer (AdamW) to update the parameters of the model during training
    optimizer = optim.AdamW(model.parameters(), eps=ADAM_EPS)

    # reload the training state from the most recent checkpoint if the -r option is specified.
    if RESTART_MODE:
        INIT_EPOCH, LAST_EPOCH, model, optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_MODEL, CHECKPOINT_OPT, N_EPOCHS, model, optimizer)
        print('')
    # prepare the label tensor that indicates the domain of the input image (0 for source domain, 1 for target domain)
    label=torch.zeros(BATCH_SIZE,dtype=torch.int64)
    label[BATCH_SIZE//2:]=torch.ones(BATCH_SIZE//2,dtype=torch.int64)
    label=label.to(DEVICE)
    
    # loss function for VAE and label discriminator
    loss_func = VAELoss(channels=C, alpha=0.01)
    loss_func_dis=nn.CrossEntropyLoss()
    loss_func_v = nn.L1Loss()
    loss_scale=10**(0) # The scale of the loss function for the label discriminator 
    # prepare the loss function value visualizer
    visualizer = LossVisualizer(['train loss','train dis loss','valid loss','valid dis loss'], init_epoch=INIT_EPOCH)
    # Training the model for the specified number of epochs
    for epoch in range(INIT_EPOCH, LAST_EPOCH):
        print('Epoch {0}:'.format(epoch + 1))

        # Training
        model.train()
        sum_loss = 0
        sum_dis_par=0
        sum_dis_loss=0
        sketch_count=0
        for X_1 in tqdm(source_train_dataloader):
            for param in model.parameters():
                param.grad = None
            # X is the input to the model (the first half of X is source domain images, and the second half is target domain images)
            X=torch.empty(BATCH_SIZE,3,H,W)
            X[0:len(X_1),:,:,:]=X_1
            batch_count=len(X_1)
            while batch_count<BATCH_SIZE: 
                X[batch_count]=target_train_dataset[sketch_count]
                batch_count+=1
                sketch_count+=1
                #if the number of target domain images used in the current epoch exceeds the total number of target domain images, start from the beginning of the target domain dataset again
                if sketch_count>=len(target_train_dataset):
                   sketch_count=0
            X = X.to(DEVICE)
            
            X = image_transform(X) # data augmentation (random horizontal flip)
            Y, mu, lnvar,dis_result = model(X,label,grl_scale=2/(1+np.exp(-10*((epoch+1)/N_EPOCHS)))-1) # input X to the model
            # calculate the current value of the loss function and optimize the model parameters
            dis_loss=loss_func_dis(dis_result,label)
            loss = loss_func(Y, X, mu, lnvar) + loss_scale*dis_loss
            LOSS_SCALER.scale(loss).backward() 
            LOSS_SCALER.step(optimizer)
            LOSS_SCALER.update() 
            # record the current value of the loss function
            sum_dis_loss += float(dis_loss)* len(X)
            sum_loss += float(loss) * len(X) 
            sum_dis_par+= torch.count_nonzero(torch.argmax(dis_result,dim=1)-label)
        # calculate the average value of the loss function at the end of the epoch
        avg_dis_par= sum_dis_par/ (2*source_train_size)
        avg_dis_loss= sum_dis_loss/ (2*source_train_size)
        avg_loss = sum_loss / (2*source_train_size)
        visualizer.add_value('train loss', avg_loss) # register the current value of the loss function with the visualizer
        visualizer.add_value('train dis loss', avg_dis_loss)
        print('train loss = {0:.6f}'.format(avg_loss))
        print('train dis_loss = {0:.6f}'.format(avg_dis_loss))
        print('train dis_par= {0:.6f} %'.format(100*(1-avg_dis_par)))
        
        # Validation
        model.eval()
        sum_loss = 0
        sum_dis_loss=0
        sketch_count=0
        sum_dis_par=0
        with torch.inference_mode():
            for X_1 in tqdm(source_valid_dataloader):
                # X is the input to the model (the first half of X is source domain images, and the second half is target domain images)
                X=torch.empty(BATCH_SIZE,3,H,W)
                X[0:len(X_1),:,:,:]=X_1
                batch_count=len(X_1)
                while batch_count<BATCH_SIZE:
                    X[batch_count]=target_valid_dataset[sketch_count]
                    batch_count+=1
                    sketch_count+=1
                    if sketch_count>=len(target_valid_dataset):
                       sketch_count=0
                X = X.to(DEVICE)
                Y,dis_result = model(X,label, testmode=True)#input X to the model and get output Y
                # calculate the current value of the loss function
                dis_loss=loss_func_dis(dis_result,label)
                loss = loss_func_v(Y, X)+loss_scale*dis_loss
                # record the current value of the loss function
                sum_loss += float(loss) * len(X)
                sum_dis_loss += float(dis_loss)*len(X)
                sum_dis_par+= torch.count_nonzero(torch.argmax(dis_result,dim=1)-label)
        # calculate the average value of the loss function at the end of the epoch
        avg_loss = sum_loss / (2*source_valid_size)
        avg_dis_loss= sum_dis_loss / (2*source_valid_size)
        avg_dis_par= sum_dis_par / (2*source_valid_size)
        # register the current value of the loss function with the visualizer
        visualizer.add_value('valid loss', avg_loss) 
        visualizer.add_value('valid dis loss', avg_dis_loss) 
        print('')
        print('valid loss(dis) = {0:.6f}'.format(avg_dis_loss))
        print('')
        print('valid dis parcent = {0:.6f} %'.format(100*(1-avg_dis_par)))
        print('')
        # The process of visualizing the training progress by displaying the original input images and the reconstructed images
        X=torch.empty(BATCH_SIZE,C,H,W)
        for i in range(BATCH_SIZE//2):
            X[i]=source_valid_dataset[i]
            X[BATCH_SIZE//2+i]=target_valid_dataset[i]
        X=X.to(DEVICE)
        Y,_= model(X,label,testmode=True)
        # display the training progress (the number of images to be displayed is specified by the argument num of the function show_images)
        if epoch == 0:
            X = to_sigmoid_image(X)
            show_images(X.to('cpu').detach(), num=BATCH_SIZE, num_per_row=8, title='original', save_only=True, save_dir=MODEL_DIR) # display the original input images(The first epoch only)
        Y = to_sigmoid_image(Y)
        show_images(Y.to('cpu').detach(), num=BATCH_SIZE, num_per_row=8, title='vae_epoch_{0}'.format(epoch + 1), save_only=True, save_dir=MODEL_DIR) # display the reconstructed images
        del Y
        torch.cuda.empty_cache()

        # display the training progress by showing the loss function value graph
        visualizer.show(sec=1) # hold on the 1 second to update the graph display
        # save the loss function value graph and the history of the loss function values in a file
        visualizer.save(v_file=os.path.join(MODEL_DIR, 'vae_loss_graph.png'), h_file=os.path.join(MODEL_DIR, 'vae_loss_history.csv'))

        # save the model parameters and the training state to files as a checkpoint at the end of each epoch
        save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_MODEL, CHECKPOINT_OPT, epoch+1, model, optimizer)

        # save the model during training automotically at the specified interval if the -s option is specified
        if AUTO_SAVE and (epoch+1) % AUTO_SAVE_INTERVAL == 0:
            shutil.copy(CHECKPOINT_MODEL, autosaved_model_name(MODEL_PATH, epoch + 1))
    # save the trained neural network model to a file at the end of the training
    torch.save(model.to('cpu').state_dict(), MODEL_PATH)
     
if __name__ == '__main__':
    main()
