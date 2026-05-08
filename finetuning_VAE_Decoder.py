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
from mylib.loss_functions import VAELoss,GANLoss
from mylib.visualizers import LossVisualizer
from mylib.data_io import CSVBasedDataset, autosaved_model_name, to_sigmoid_image, show_images
from mylib.utility import print_args, save_datasets, load_datasets_from_file, save_checkpoint, load_checkpoint
from networks import  Discriminator
from mylib.loss_functions import GANLoss

# the file path of Datasets
SOURCE_DATASET_CSV = './Dataset/crop_face_images.csv'
TARGET_DATASET_CSV = './Dataset/crop_sketches_100.csv'
# The path of directory that include image files
DATA_DIR = './Dataset/'
LABEL_EMBED_DIM=64
TARGET_NUM=100
# the folder to save results of training
MODEL_DIR = './FINETUNING_model'

# the size and channels of images
C = 3 # channel
H = 512 # height
W = 512 # width
N = 2 #the number of class
# the channels of feature maps
ZC = 4

# the interval of epoch to save the model if auto-save mode is available.
AUTO_SAVE_INTERVAL = 10

ATTRIBUTES=['Real_face','Sketch_face']
# check point file to restart training the model
CHECKPOINT_EPOCH = os.path.join(MODEL_DIR, 'checkpoint_epoch.pkl')
CHECKPOINT_GEN_MODEL = os.path.join(MODEL_DIR, 'checkpoint_gen_model.pth')
CHECKPOINT_SOURCE_DIS_MODEL = os.path.join(MODEL_DIR, 'checkpoint_source_dis_model.pth')
CHECKPOINT_TARGET_DIS_MODEL = os.path.join(MODEL_DIR, 'checkpoint_target_dis_model.pth')
CHECKPOINT_GEN_OPT = os.path.join(MODEL_DIR, 'checkpoint_gen_opt.pth')
CHECKPOINT_SOURCE_DIS_OPT = os.path.join(MODEL_DIR, 'checkpoint_source_dis_opt.pth')
CHECKPOINT_TARGET_DIS_OPT = os.path.join(MODEL_DIR, 'checkpoint_target_dis_opt.pth')

# the loss function of L_{dist}
class DiversityLoss(nn.Module):

    def __init__(self, lambda_ = 10000):
        super(DiversityLoss, self).__init__()
        self.lambda_ = lambda_
        self.loss_func = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def cosine_sim(self, h):
        B = h.size()[0]
        h = h.reshape((B, -1))
        h = h / torch.norm(h, dim=1, keepdim=True)
        h = torch.mm(h, torch.transpose(h, 0, 1))
        h = h.flatten()[1:].view(B-1, B+1)[:, :-1].reshape(B, B-1)
        return h

    def forward(self, h_learned, h_fixed):
        h_learned = nn.functional.log_softmax(self.cosine_sim(h_learned), dim=1)
        h_fixed = nn.functional.log_softmax(self.cosine_sim(h_fixed), dim=1)
        loss = self.loss_func(h_learned, h_fixed)
        return self.lambda_ * loss


def main():
    
    # get the Device , the number of epochs and batch size. etc. from argument and save to the variable
    parser = argparse.ArgumentParser(description='Latent Denoising Diffusion Probabilistic Model Sample Code (VAE training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=16, type=int, help='minibatch size')
    parser.add_argument('--generator_model', '-gm', default=os.path.join(MODEL_DIR, 'gen_model.pth'), type=str, help='file path of generator model')
    parser.add_argument('--discriminator_model', '-dm', default=os.path.join(MODEL_DIR, 'source_dis_model.pth'), type=str, help='file path of discriminator model')
    parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'vae_model.pth'), type=str, help='file path of trained model')
    parser.add_argument('--autosave', '-s', help='this option makes the model automatically saved in each epoch', action='store_true')
    parser.add_argument('--restart', '-r', help='this option makes the training proccess restart from the most recent checkpoint', action='store_true')
    args = print_args(parser.parse_args())
    DEVICE = args['device']
    N_EPOCHS = args['epochs']
    BATCH_SIZE = args['batchsize']
    MODEL_PATH = args['model']
    AUTO_SAVE = args['autosave']
    GEN_MODEL_PATH = args['generator_model']
    SOURCE_DIS_MODEL_PATH = os.path.join(MODEL_DIR, 'source_dis_model.pth')
    TARGET_DIS_MODEL_PATH = os.path.join(MODEL_DIR, 'target_dis_model.pth')
    RESTART_MODE = args['restart']

    INIT_EPOCH = 0 # the first epoch number
    LAST_EPOCH = INIT_EPOCH + N_EPOCHS # the final epoch number 

        # load csv file of dataset(SOURCE_DOMAIN)
    source_dataset = CSVBasedDataset(
            filename = SOURCE_DATASET_CSV,
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
        # load csv file of dataset(TARGET_DOMAIN)
    target_dataset = CSVBasedDataset(
            filename = TARGET_DATASET_CSV,
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
    # prepare data loaders to use the training datasets in mini-batches
    source_train_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE//2, shuffle=True, pin_memory=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE//2, shuffle=True, pin_memory=True)

    # create the generator model(pre-trained VAE decoder by source domain dataset) and move it to the specified device (GPU or CPU)
    gen_model = LDDPM_VAE(C=C, ZC=ZC,label_embed_dim=LABEL_EMBED_DIM).to(DEVICE)
    gen_model.load_state_dict(torch.load('./models/real_face_vae_model.pth'))
    gen_model=gen_model.to(DEVICE)
    # create the fixed generator model(pre-trained VAE decoder by source domain dataset) to caliculate L_{dist} loss
    gen_model_fixed = LDDPM_VAE(C=C,ZC=ZC,label_embed_dim=LABEL_EMBED_DIM).to(DEVICE)
    gen_model_fixed.load_state_dict(torch.load('./models/real_face_vae_model.pth'))
    gen_model_fixed=gen_model_fixed.to(DEVICE)
    gen_model_fixed.eval()
    
    # create the discriminator model for both source and target domain.
    target_dis_model = Discriminator(C=C, H=H, W=W).to(DEVICE)
    source_dis_model=Discriminator(C=C, H=H, W=W).to(DEVICE)
    # create the optimizer (AdamW) to update the parameters of the model during training
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    target_dis_optimizer = optim.Adam(target_dis_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    source_dis_optimizer = optim.Adam(source_dis_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # if -r option is specified, reload the training status from checkpoint files and restart to train the model
    if RESTART_MODE:
        INIT_EPOCH, LAST_EPOCH, gen_model, gen_optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_GEN_MODEL, CHECKPOINT_GEN_OPT, N_EPOCHS, gen_model, gen_optimizer)
        _, _, target_dis_model, target_dis_optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_TARGET_DIS_MODEL, CHECKPOINT_TARGET_DIS_OPT, N_EPOCHS, target_dis_model, target_dis_optimizer)
        _, _, source_dis_model, source_dis_optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_SOURCE_DIS_MODEL, CHECKPOINT_SOURCE_DIS_OPT, N_EPOCHS, source_dis_model, source_dis_optimizer)
        
        print('')
    # feature vector(real)
    real_label=torch.zeros(BATCH_SIZE,dtype=torch.int64)
   
    real_label=real_label.to(DEVICE)
    # feature vector(fake)
    fake_label=torch.ones(BATCH_SIZE,dtype=torch.int64)
    fake_label[BATCH_SIZE//2:]=torch.zeros(BATCH_SIZE//2,dtype=torch.int64)
    fake_label=fake_label.to(DEVICE)
    
    # loss function
    loss_func_Ldist = DiversityLoss(lambda_ = 10000) 
    loss_func =GANLoss(label_smoothing=True)# GAN Loss
    
    # prepare the visualizer for loss function
    visualizer = LossVisualizer(['G loss', 'D loss'], init_epoch=INIT_EPOCH)
    # prepare the latent-features for validation
    target_count=0
    seed = np.arange(BATCH_SIZE)
    z = [np.loadtxt('./Dataset/face_img_features/{}.txt'.format(i)).reshape((4,64,64)) for i in seed]
    z_reshape = np.asarray(z)
    Z_valid=torch.from_numpy(z_reshape.astype(np.float32)).clone()
    Z_valid=Z_valid.to(DEVICE)
    # Zanch(real face,sketch face)
    mu=np.loadtxt(f'./Dataset/face_img_features/1000.txt')
    z_anch=mu.reshape((4,64,64))
    for i in range(1001,1000+BATCH_SIZE//2):
        comp_path=f'./Dataset/face_img_features/{i}.txt'
        mu=np.loadtxt(comp_path)
        mu_reshape=mu.reshape((4,64,64))
        z_anch=np.append(z_anch,mu_reshape)
    for i in range(BATCH_SIZE//2):
        comp_path=f'./Dataset/sketch_features/{i}.txt'
        mu=np.loadtxt(comp_path)
        mu_reshape=mu.reshape((4,64,64))
        z_anch=np.append(z_anch,mu_reshape)
    z_anch_reshape=z_anch.reshape((BATCH_SIZE,4,64,64))
    Z_anch_origin=torch.from_numpy(z_anch_reshape.astype(np.float32)).clone()
    # Training
    for epoch in range(INIT_EPOCH, LAST_EPOCH):
        print('Epoch {0}:'.format(epoch + 1))
        target_count=0
        sum_gen_loss=0
        sum_dis_loss=0
        gen_model.train() 
        source_dis_model.train() 
        target_dis_model.train()

        for real_source,real_target in tqdm(zip(source_train_dataloader,target_train_dataloader)):
            for param in gen_model.parameters():
                param.grad = None
            for param in target_dis_model.parameters():
                param.grad = None
            for param in source_dis_model.parameters():
                param.grad = None
            #preparing source featuremaps
            seed = np.random.randint(0, 69954, size=BATCH_SIZE//2)
            z_source = [np.loadtxt('./Dataset/face_img_features/{}.txt'.format(i)).reshape((4,64,64)) for i in seed]
            z_reshape_source = np.asarray(z_source)
            if target_count+BATCH_SIZE//2>TARGET_NUM:
                target_count=TARGET_NUM-BATCH_SIZE//2
            z_target = [np.loadtxt('./Dataset/sketch_features/{}.txt'.format(i)).reshape((4,64,64)) for i in range(target_count,target_count+BATCH_SIZE//2)]
            z_reshape_target=np.asarray(z_target)
            target_count+=BATCH_SIZE//2
            z_reshape=np.concatenate((z_reshape_source,z_reshape_target),axis=0)
            Z=torch.from_numpy(z_reshape.astype(np.float32)).clone()
            Z=Z.to(DEVICE) # prepare latent features Z (the first half of Z is source doamin, the second falf of Z is target domain)
            Z_anch=Z_anch_origin+0.5*torch.randn_like(Z_anch_origin)
            Z_anch=Z_anch.to(DEVICE)
            real_source=real_source.to(DEVICE)
            real_target=real_target.to(DEVICE)
            

            _, h4f, h3f, h2f, h1f = gen_model_fixed.decode_with_hidden_outputs(Z,fake_label)
            fake, h4, h3, h2, h1 = gen_model.decode_with_hidden_outputs(Z,fake_label)
            fake_whole = gen_model.decode(Z_anch,fake_label) # to discriminate using all part of images
            fake_cpy = fake.detach() # Copy of Fake_img
            fake_whole_cpy = fake_whole.detach() # Copy of Fake_img
            _, presul_fake_target_patch = target_dis_model(fake[0:BATCH_SIZE//2]) # Discriminate Fake target domain images
            _, presul_fake_source_patch = source_dis_model(fake[BATCH_SIZE//2:]) # Discriminate Fake source domain images
            presul_fake_target , _ = target_dis_model(fake_whole[0:BATCH_SIZE//2])
            presul_fake_source,_= source_dis_model(fake_whole[BATCH_SIZE//2:])
            
            # calculate the loss value and update the generator model(VAE decoder)
            gen_loss = loss_func.G_loss(presul_fake_target) + loss_func.G_loss(presul_fake_target_patch) +loss_func_Ldist(h4, h4f) + loss_func_Ldist(h3, h3f) +loss_func_Ldist(h2, h2f) + loss_func_Ldist(h1, h1f) 
            gen_loss += loss_func.G_loss(presul_fake_source)+loss_func.G_loss(presul_fake_source_patch)
            gen_loss.backward() 
            gen_optimizer.step()
            #training of Discriminator
            for param in source_dis_model.parameters():
               param.grad = None 
            for param in target_dis_model.parameters():
               param.grad = None 
            presul_real_target, presul_real_target_patch = target_dis_model(real_target)# Discriminate Real target domain images
            presul_real_source, presul_real_source_patch = source_dis_model(real_source) # Discriminate Real source domain images
            
            _, presul_fake_target_patch = target_dis_model(fake_cpy[0:BATCH_SIZE//2]) # Discriminate fake target domain images 
            _, presul_fake_source_patch = source_dis_model(fake_cpy[BATCH_SIZE//2:]) # Discriminate fake source domain images
            presul_fake_target,_= target_dis_model(fake_whole_cpy[0:BATCH_SIZE//2])
            presul_fake_source,_= source_dis_model(fake_whole_cpy[BATCH_SIZE//2:])
            
            target_dis_loss = loss_func.D_loss(presul_fake_target, as_real=False) + loss_func.D_loss(presul_real_target, as_real=True) + loss_func.D_loss(presul_fake_target_patch, as_real=False) + loss_func.D_loss(presul_real_target_patch, as_real=True)
            source_dis_loss = loss_func.D_loss(presul_fake_source, as_real=False) + loss_func.D_loss(presul_real_source, as_real=True) + loss_func.D_loss(presul_fake_source_patch, as_real=False) + loss_func.D_loss(presul_real_source_patch, as_real=True)
            
            target_dis_loss.backward()
            source_dis_loss.backward()
            target_dis_optimizer.step()
            source_dis_optimizer.step()
            
          
            sum_gen_loss += float(gen_loss)*len(Z)
            sum_dis_loss += float(target_dis_loss)*len(Z)
        
       
        avg_gen_loss = sum_gen_loss / (len(target_dataset)*2)
        avg_dis_loss = sum_dis_loss / (len(target_dataset)*2)
        # record the current value of the loss function
        visualizer.add_value('G loss', avg_gen_loss) 
        visualizer.add_value('D loss', avg_dis_loss) 
        
        print('train gen_loss= {0:.6f} '.format(avg_gen_loss))
        print('train dis_loss= {0:.6f} '.format(avg_dis_loss))
        
        # Validation
        gen_model.eval()
        with torch.inference_mode():
            Y= gen_model.decode(Z_valid,torch.ones(BATCH_SIZE,dtype=torch.int64).to(DEVICE))
        # 学習経過の表示（表示する画像の枚数は関数 show_images の引数 num で指定）
            if epoch == 0:
                Y_valid = gen_model_fixed.decode(Z_valid,real_label)
                Y_valid = to_sigmoid_image(Y_valid)
                show_images(Y_valid.to('cpu').detach(), num=BATCH_SIZE, num_per_row=8, title='original', save_only=True, save_dir=MODEL_DIR) # display input images (only first epoch)
            Y = to_sigmoid_image(Y)
            show_images(Y.to('cpu').detach(), num=BATCH_SIZE, num_per_row=8, title='vae_epoch_{0}'.format(epoch + 1), save_only=True, save_dir=MODEL_DIR) # display reconstruct images 
            del Y
            torch.cuda.empty_cache()

        # display viusalized result
        visualizer.show(sec=1) 
        # save the loss function values and visualized result
        visualizer.save(v_file=os.path.join(MODEL_DIR, 'vae_loss_graph.png'), h_file=os.path.join(MODEL_DIR, 'vae_loss_history.csv'))

        # save the chekpoint file 
        save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_GEN_MODEL, CHECKPOINT_GEN_OPT, epoch+1, gen_model, gen_optimizer)
        save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_SOURCE_DIS_MODEL, CHECKPOINT_SOURCE_DIS_OPT, epoch+1, source_dis_model, source_dis_optimizer)
        save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_TARGET_DIS_MODEL, CHECKPOINT_TARGET_DIS_OPT, epoch+1, target_dis_model, target_dis_optimizer)
        # save the model if auto save mode is specified
        if AUTO_SAVE and (epoch+1) % AUTO_SAVE_INTERVAL == 0:
            shutil.copy(CHECKPOINT_GEN_MODEL, autosaved_model_name(GEN_MODEL_PATH, epoch + 1))
            shutil.copy(CHECKPOINT_TARGET_DIS_MODEL, autosaved_model_name(TARGET_DIS_MODEL_PATH, epoch + 1))
            shutil.copy(CHECKPOINT_SOURCE_DIS_MODEL, autosaved_model_name(SOURCE_DIS_MODEL_PATH, epoch + 1))
    # save the training model both generator(VAE decoder) and discriminator(source and target)
    torch.save(gen_model.to('cpu').state_dict(), GEN_MODEL_PATH)
    torch.save(target_dis_model.to('cpu').state_dict(), TARGET_DIS_MODEL_PATH)
    torch.save(source_dis_model.to('cpu').state_dict(),SOURCE_DIS_MODEL_PATH)
     
if __name__ == '__main__':
    main()
