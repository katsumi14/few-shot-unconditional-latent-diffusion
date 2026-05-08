import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lighten(c):
    return np.uint8(255 - (255 - c) // 2)

# The visualizer for two dimension data discriminator
class ClassifierVisualizer():

    # Constructor
    #   - n_classes: the number of class 
    #   - size: vislualized image's size
    #   - hrange: the range of horizon axis(min,max)
    #   - vrange: the range of vertical axis(min,max)
    #   - title: the title of visualized graph
    #   - hlabel: horizon axis label name
    #   - vlabel: vertical axis label name 
    #   - clabels: class label name
    #   - bins: the resolution of horizon・vertical axis
    def __init__(self, n_classes, size=512, hrange=(-1.0, 1.0), vrange=(-1.0, 1.0), title='title', hlabel='feature 1', vlabel='feature 2', clabels=None, bins=5):
        self.n_classes = n_classes
        self.size = size
        self.bins = bins
        self.hrange = hrange
        self.vrange = vrange
        self.title = title
        self.hlabel = hlabel
        self.vlabel = vlabel
        if clabels is None:
            self.clabels = []
            for i in range(n_classes):
                self.clabels.append('class {0}'.format(i + 1))
        else:
            self.clabels = clabels
        for i in range(0, size):
            y = np.ones((size, 1), dtype=np.float32) * i
            x = np.asarray([np.arange(size)], dtype=np.float32).transpose(1, 0)
            y = vrange[0] + y * (vrange[1] - vrange[0]) / (size - 1)
            x = hrange[0] + x * (hrange[1] - hrange[0]) / (size - 1)
            c = np.concatenate([x, y], axis=1)
            self.data = c if i == 0 else np.concatenate([self.data, c], axis=0)

    # Visualize
    #   - model: the instance of discriminator class
    #   - class_colors: the color of each class(RGB)
    #   - sec: the time of displaying the result(if less 1 sec, set it as 1 sec. default is 1 sec)
    #   - samples: the data to visualize the graph
    #              samples[0] is label data (numpy.ndarray, int32), 
    #              samples[1] is feature data(numpy.ndarray, float32)
    #   - other args: please refer constructor, and change them if you need
    def show(self, model, class_colors, sec=1, samples=None, title=None, hlabel=None, vlabel=None, clabels=None, bins=None):

        for param in model.parameters():
            device = param.data.device
            break

        plt.cla()

        # if parameter is changed, they are updated
        if title is not None: self.title = title
        if hlabel is not None: self.hlabel = hlabel
        if vlabel is not None: self.vlabel = vlabel
        if clabels is not None: self.clabels = clabels
        if bins is not None: self.bins = bins

        # set the graph title
        plt.title(self.title)

        # make background image
        model.eval()
        result = torch.argmax(model(torch.tensor(self.data, device=device)), dim=1)
        result = result.to('cpu').detach().numpy().copy().reshape((self.size, self.size, 1))
        r = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        g = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        b = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        for i in range(self.n_classes):
            r += np.where(result == i, lighten(class_colors[i][0]), np.uint8(0))
            g += np.where(result == i, lighten(class_colors[i][1]), np.uint8(0))
            b += np.where(result == i, lighten(class_colors[i][2]), np.uint8(0))
        img = np.concatenate([r, g, b], axis=2)
        del result

        # set the background image
        plt.imshow(img)

        # plot the data
        if samples is not None:
            lab = np.asarray(samples[1])
            feat = np.asarray(samples[0])
            ptx = np.floor((self.size - 1) * (feat[:,0] - self.hrange[0]) / (self.hrange[1] - self.hrange[0])).astype(np.int32)
            pty = np.floor((self.size - 1) * (feat[:,1] - self.vrange[0]) / (self.vrange[1] - self.vrange[0])).astype(np.int32)
            for i in range(self.n_classes):
                ccolor = np.asarray(class_colors[i]).reshape((1, 3)) / 255
                plt.scatter(ptx[lab==i], pty[lab==i], c=ccolor, label=self.clabels[i])
            plt.legend(loc='best')

        # make the vertical and horizon axis's scale
        hlabels = []
        vlabels = []
        for i in range(0, self.bins):
            hlabels.append(format(self.hrange[0] + i * (self.hrange[1] - self.hrange[0]) / (self.bins - 1), '.3g'))
            vlabels.append(format(self.vrange[0] + i * (self.vrange[1] - self.vrange[0]) / (self.bins - 1), '.3g'))
        plt.xticks(np.linspace(0, self.size-1, self.bins), hlabels)
        plt.yticks(np.linspace(0, self.size-1, self.bins), vlabels)
        plt.grid()

        # set the label of axis
        plt.xlabel(self.hlabel)
        plt.ylabel(self.vlabel)

        # set the vertical axis upside down
        plt.gca().invert_yaxis()

        # display (when 1 sec passed, the window is closed)
        plt.pause(sec)


# visualizer of the loss function's value
class LossVisualizer():

    # Constructor
    #   - items: the list of loss's name
    #   - log_mode: whether the scale set the log or not.
    #   - init_epoch: first epoch number
    def __init__(self, items, log_mode=False, init_epoch=0):
        self.init_epoch = init_epoch + 1
        self.log_mode = log_mode
        self.loss_values = {}
        for item in items:
            if not item in self.loss_values.keys():
                self.loss_values[item] = np.empty(0)

    # add the value
    #   - item: the loss's name 
    #   - value: the loss value
    def add_value(self, item, value):
        if item in self.loss_values.keys():
            self.loss_values[item] = np.append(self.loss_values[item], value)

    # Visualize
    #   - sec: display time(if less than 1 sec, set it 1 sec. default is 1 sec)
    def show(self, sec=1):
        plt.cla()
        plt.title('Loss history')
        plt.xlabel('epoch')
        if self.log_mode:
            plt.yscale('log')
        plt.ylabel('loss value')
        plt.grid()
        for item in self.loss_values.keys():
            t = np.arange(self.init_epoch, len(self.loss_values[item]) + self.init_epoch)
            plt.plot(t, self.loss_values[item], label=item)
        plt.legend()
        plt.pause(sec)

    # save the log of loss function and visualized result
    #   - v_file: the file to save the visulaized result
    #   - h_file: the file to save the loss function log
    def save(self, v_file, h_file):
        plt.cla()
        plt.title('Loss history')
        plt.xlabel('epoch')
        if self.log_mode:
            plt.yscale('log')
        plt.ylabel('loss value')
        plt.grid()
        for item in self.loss_values.keys():
            t = np.arange(self.init_epoch, len(self.loss_values[item]) + self.init_epoch)
            plt.plot(t, self.loss_values[item], label=item)
        plt.legend()
        plt.savefig(v_file)
        df = pd.DataFrame(self.loss_values, columns=self.loss_values.keys())
        df.reset_index(drop=True, inplace=True)
        df.index = np.arange(self.init_epoch, len(df) + self.init_epoch)
        df.to_csv(h_file)
