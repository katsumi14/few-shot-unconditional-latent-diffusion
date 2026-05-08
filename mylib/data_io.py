import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib import cm
from torch.utils.data import Dataset



# change i to one-hot vector of length n
def __to_one_hot__(i, n):
    a = np.zeros(n, dtype=np.int32)
    a[i] = 1
    return a


# The function converts a list of categorical data into a list of numerical data(np.int32)
def __to_numerical__(cat_data, one_hot=False, fdict=None):

    # select the number of kinds of data values
    s = sorted(set(cat_data))
    n = len(s)

    # make a drectory for forward and reverse lookup
    if fdict is None:
        if one_hot:
            forward_dict = {item:__to_one_hot__(i, n) for i, item in enumerate(s)}
        else:
            forward_dict = {item:i for i, item in enumerate(s)}
    else:
        forward_dict = fdict
    reverse_dict = [s[i] for i in range(n)]

    # Convert
    data = np.asarray(list(map(forward_dict.get, cat_data)), dtype=np.int32)

    return data, forward_dict, reverse_dict



# detect the number that specified by target_indices from the array arr
def __extract__(arr, target_indices):
    return_value = [item for i, item in enumerate(arr) if i in target_indices]
    if type(arr) == np.ndarray:
        return np.asarray(return_value, dtype=arr.dtype)
    elif type(arr) == torch.tensor:
        return torch.tensor(return_value, dtype=arr.dtype, device=arr.device)
    else:
        return return_value



# The class for loading dataset from csv file
#   -filename: the file path of csv file to load
#   -items: the name of columns to load as data
#   -dtypes: the data type of each column (one of 'float', 'label', 'one-hot', 'image')
#   -target_indices: the set of indices to load (if None, load all data. Default is None)
#   -fdicts: the forward lookup dictionaries for converting categorical data into numerical data (if None, automatically created)
#   -dirname: used for specifying the directory name to be added to the beginning of file names when the data type is 'image'
#   -img_mode: used for specifying whether the image is color or not when the data type is 'image' ('color' or 'grayscale')
#   -img_range: used for specifying the range of pixel values to normalize when the data type is 'image' (expected to be [0, 1] or [-1, 1], but other ranges can also be specified)
#   -img_transform: used for specifying the Transform object to be used as preprocessing when the data type is 'image'
class CSVBasedDataset(Dataset):

    def __init__(self, filename, items, dtypes, target_indices=None, fdicts=None, dirname='./', img_mode='', img_range=[0, 1], img_transform=None):
        super(CSVBasedDataset, self).__init__()

        self.dtypes = dtypes
        self.dirname = dirname
        self.img_range = img_range
        self.img_range_coeff = (self.img_range[1] - self.img_range[0]) / 255
        self.img_transform = img_transform
        if img_mode == 'color':
            self.img_mode = torchvision.io.image.ImageReadMode.RGB
        elif img_mode == 'grayscale':
            self.img_mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            self.img_mode = torchvision.io.image.ImageReadMode.UNCHANGED

        df = pd.read_csv(filename)


        # save the data of specified columns in items as member variables
        self.data = []
        self.forward_dicts = []
        self.reverse_dicts = []
        for i in range(len(items)):
            fd = None
            rd = None
            if dtypes[i] == 'float':
                X = torch.tensor(df[items[i]].values, dtype=torch.float32, device='cpu')
            elif dtypes[i] == 'label':
                if type(items[i]) is list:
                    X = []
                    fd = []
                    rd = []
                    j = 0
                    for item in items[i]:
                        if fdicts is None:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=False)
                        else:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=False, fdict=fdicts[i][j])
                        X.append(X_temp)
                        fd.append(fd_temp)
                        rd.append(rd_temp)
                        j += 1
                    X = np.concatenate(X, axis=1)
                else:
                    if fdicts is None:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=False)
                    else:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=False, fdict=fdicts[i])
                X = torch.tensor(X, dtype=torch.long, device='cpu')
            elif dtypes[i] == 'one-hot':
                if type(items[i]) is list:
                    X = []
                    fd = []
                    rd = []
                    j = 0
                    for item in items[i]:
                        if fdicts is None:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=True)
                        else:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=True, fdict=fdicts[i][j])
                        X.append(X_temp)
                        fd.append(fd_temp)
                        rd.append(rd_temp)
                        j += 1
                    X = np.concatenate(X, axis=1)
                else:
                    if fdicts is None:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=True)
                    else:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=True, fdict=fdicts[i])
                X = torch.tensor(X, dtype=torch.float32, device='cpu')
            elif dtypes[i] == 'image':
                X = df[items[i]].to_list()
            else:
                continue
            if target_indices is not None:
                X = __extract__(X, target_indices)
            self.data.append(X)
            self.forward_dicts.append(fd)
            self.reverse_dicts.append(rd)

        # save the dataset size as a member variable "len" of type int
        self.len = len(self.data[0])

    # the function returns the dataset size
    def __len__(self):
        return self.len

    # The function returns the index-th data 
    # data loader calls this function as many times as needed to automatically create mini-batches
    def __getitem__(self, index):
        single_data = []
        for i in range(len(self.data)):
            if self.dtypes[i] == 'image':
                # load image files and normalize pixel value
                x = torchvision.io.read_image(os.path.join(self.dirname, self.data[i][index]), mode=self.img_mode)
                x = x * self.img_range_coeff + self.img_range[0]
                if self.img_transform is not None:
                    x = self.img_transform(x)
            else:
                x = self.data[i][index]
            single_data.append(x)
        if len(self.data) == 1:
            return single_data[0]
        else:
            return tuple(single_data)



# The class for loading image data as triplets
#  -filename: the file path of csv file to load
#  -data_item: the name of column to load as data (the patho of image files)
#  -label_item: the name of column to load as label (the object class name, person ID, etc.)
#  -target_indices: the set of indices to load (if None, load all data. Default is None)
#  -target_labels: the set of labels to load (used for separating training data and validation data by label. Default is ignored)
#  -use_anchor_label: whether to use the label information of anchor data at the same time or not
#  -fdict: the forward lookup dictionary for converting labels into integer values (if None, automatically created)
#  -dirname: used for specifying the directory name to be added to the beginning of file names when the data type is 'image'
#  -img_mode: used for specifying whether the image is color or not when the data type is 'image' ('color' or 'grayscale')
#  -img_range: used for specifying the range of pixel values to normalize when the data type is 'image' (expected to be [0, 1] or [-1, 1], but other ranges can also be specified)
#  -img_transform: used for specifying the Transform object to be used as preprocessing when the data type is 'image' 
class TripletImageDataset(Dataset):

    # コンストラクタ
    def __init__(self, filename, data_item, label_item, target_indices=None, target_labels=None, use_anchor_label=False, fdict=None, dirname='./', img_mode='', img_range=[0, 1], img_transform=None):
        super(TripletImageDataset, self).__init__()

        self.use_anchor_label = use_anchor_label
        self.dirname = dirname
        self.img_range = img_range
        self.img_range_coeff = (self.img_range[1] - self.img_range[0]) / 255
        self.img_transform = img_transform
        if img_mode == 'color':
            self.img_mode = torchvision.io.image.ImageReadMode.RGB
        elif img_mode == 'grayscale':
            self.img_mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            self.img_mode = torchvision.io.image.ImageReadMode.UNCHANGED

        df = pd.read_csv(filename)

        # get the list of image file names and the list of labels
        if target_labels is None:
            self.data = df[data_item].to_list()
            self.label = df[label_item].to_list()
            if fdict is None:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False)
            else:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False, fdict=fdict)
        else:
            temp_data = df[data_item].to_list()
            temp_label = df[label_item].to_list()
            self.data = [ temp_data[i] for i in range(len(temp_label)) if temp_label[i] in target_labels ]
            self.label = [ temp_label[i] for i in range(len(temp_label)) if temp_label[i] in target_labels ]
            if fdict is None:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False)
            else:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False, fdict=fdict)
            del temp_data
            del temp_label
        if target_indices is not None:
            self.data = __extract__(self.data, target_indices)
            self.label = __extract__(self.label, target_indices)

        # save the dataset size as a member variable "len" of type int
        self.len = len(self.data)

    # The function returns the dataset size
    def __len__(self):
        return self.len

    # The function returns the index-th data
    # data loader calls this function as many times as needed to automatically create mini-batches
    def __getitem__(self, index):

        # randomly select positive and negative samples for the anchor image at the index-th position
        lab = self.label[index]
        p_index = index # 同じラベルの画像が 1 枚しかない場合は，仕方がないので anchor == positive を許容する
        p_cands = np.where(self.label == lab)[0]
        if len(p_cands) >= 2:
            while p_index == index:
                p_index = np.random.choice(p_cands)
        n_index = np.random.choice(np.where(self.label != lab)[0])

        # load the image files and normalize pixel values
        anc = torchvision.io.read_image(os.path.join(self.dirname, self.data[index]), mode=self.img_mode)
        pos = torchvision.io.read_image(os.path.join(self.dirname, self.data[p_index]), mode=self.img_mode)
        neg = torchvision.io.read_image(os.path.join(self.dirname, self.data[n_index]), mode=self.img_mode)
        anc = anc * self.img_range_coeff + self.img_range[0]
        pos = pos * self.img_range_coeff + self.img_range[0]
        neg = neg * self.img_range_coeff + self.img_range[0]
        if self.img_transform is not None:
            anc = self.img_transform(anc)
            pos = self.img_transform(pos)
            neg = self.img_transform(neg)

        if self.use_anchor_label:
            return anc, pos, neg, torch.tensor(lab, dtype=torch.long)
        else:
            return anc, pos, neg


# The function for displaying a single image
#   - data: the image data to be displayed (expected to be normalized to the range [0, 1])
#   - title: the title of the display window
#   - sec: how many seconds to display the image (if 0 or less, the image will be displayed until the "close" button is pressed, default is 0)
#   - save_fig: if True, the displayed image will also be saved to a file (default is False)
#   - save_only: if True, the image will only be saved to a file and not displayed (default is False)
#   - save_dir: the directory to save the file if the image is saved (default is the current directory of the program)
def show_single_image(data, title='no_title', sec=0, save_fig=False, save_only=False, save_dir='./'):
    img = np.asarray(data) # change the type of input data(array) to numpy.ndarray
    if len(img.shape) == 4:
        img = img[0].transpose(1, 2, 0) #if the dimension of input data is 4, extract just one images.
    elif len(img.shape) == 3:
        if img.shape[0] == 1 or img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    elif len(img.shape) == 2:
        img = img.reshape((*img.shape, 1)) #if the dimension of input data is 2, change to three dimension's data(channel:1).
    img = (255 * np.minimum(np.ones(img.shape), np.maximum(np.zeros(img.shape), img))).astype(np.uint8) # normalize the pixel value to [0,1], and make it 255 times.
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    if save_fig or save_only:
        plt.savefig(os.path.join(save_dir, title + '.png'), bbox_inches='tight')
    if not save_only:
        if sec <= 0:
            plt.show()
        else:
            plt.pause(sec)
    plt.close()


# The function to display some images at once
#   -data: the target data to display (4 dimension of batchsize x channel x height x widht, the pixel value is normarized to [0,1])
#   -num: the number of images to display
#   -num_per_row: the number of images in row(the default is about "sqrt(num)")
#   -title: the title of display window
#   -sec: how long display(defalut is 0 sec, if 0 or less, the image will be displayed until the "close" button is pressed)
#   -save_fig: if True, save the displayed images to the file(defalut is False)
#   -save_only: if True, only save the images and not display(defalut is False)
#   -save_dir: the directory to save the file if the image is saved (default is the current directory of the program)
def show_images(data, num, num_per_row=0, title='no_title', sec=0, save_fig=False, save_only=False, save_dir='./'):
    if num_per_row <= 0:
        num_per_row = int(np.ceil(np.sqrt(num)))
    data = np.asarray(data)
    data = (255 * np.minimum(np.ones(data.shape), np.maximum(np.zeros(data.shape), data))).astype(np.uint8)
    n_total = min(data.shape[0], num) # the total number of save data
    n_rows = int(np.ceil(n_total / num_per_row)) #the number of rows to display images
    plt.figure(title, figsize=(1 + num_per_row * data.shape[3] / 128, 1 + n_rows * data.shape[2] / 128))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    for i in range(0, n_total):
        plt.subplot(n_rows, num_per_row, i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    if save_fig or save_only:
        plt.savefig(os.path.join(save_dir, title + '.png'), bbox_inches='tight')
    if not save_only:
        if sec <= 0:
            plt.show()
        else:
            plt.pause(sec)
    plt.close()



# change the pixel value's ragnge from [0,1] to [-1,1]
def to_tanh_image(img):
    return 2 * img - 1


# change the pixel value's range from [-1,1] to [0,1]
def to_sigmoid_image(img):
    return 0.5 * (img + 1)


# adjust the file name to save the training model
#   - base: the basic file name
#   - ep: epoch number
def autosaved_model_name(base: str, ep: int):
    p = base.rfind('.')
    return '{0}_ep{1}.pth'.format(base[:p], ep)
