import os
import sys
import torch
import pickle


def print_args(args):
    # change the type of comand line's argument data to dic type
    args = vars(args)

    # output the arugments
    for key, value in args.items():
        if key == 'gpu':
            dev_str = 'cuda:{0}'.format(value) if torch.cuda.is_available() and value >= 0 else 'cpu'
            print('device: {0}'.format(dev_str), file=sys.stderr)
        else:
            print('{0}: {1}'.format(key, value), file=sys.stderr)
    print('')

    if 'gpu' in args.keys():
        args['device'] = dev_str

    return args


def save_datasets(dir_, source_train_dataset, source_valid_dataset=None):
    with open(os.path.join(dir_, 'source_train_dataset.pkl'), 'wb') as f:
        pickle.dump(source_train_dataset, f)
    if source_valid_dataset is not None:
        with open(os.path.join(dir_, 'source_valid_dataset.pkl'), 'wb') as f:
            pickle.dump(source_valid_dataset, f)
def save_sub_datasets(dir_, target_train_dataset, target_valid_dataset=None):
    with open(os.path.join(dir_, 'target_train_dataset.pkl'), 'wb') as f:
        pickle.dump(target_train_dataset, f)
    if target_valid_dataset is not None:
        with open(os.path.join(dir_, 'target_valid_dataset.pkl'), 'wb') as f:
            pickle.dump(target_valid_dataset, f)


def load_datasets_from_file(dir_):
    train_dataset_file = os.path.join(dir_, 'source_train_dataset.pkl')
    valid_dataset_file = os.path.join(dir_, 'source_valid_dataset.pkl')
    if os.path.isfile(train_dataset_file):
        with open(train_dataset_file, 'rb') as f:
            source_train_dataset = pickle.load(f)
        print('{} has been loaded.'.format(train_dataset_file))
    else:
        train_dataset = None
    if os.path.isfile(valid_dataset_file):
        with open(valid_dataset_file, 'rb') as f:
            source_valid_dataset = pickle.load(f)
        print('{} has been loaded.'.format(valid_dataset_file))
    else:
        valid_dataset = None
    return source_train_dataset, source_valid_dataset
def load_sub_datasets_from_file(dir_):
    train_dataset_file = os.path.join(dir_, 'target_train_dataset.pkl')
    valid_dataset_file = os.path.join(dir_, 'target_valid_dataset.pkl')
    if os.path.isfile(train_dataset_file):
        with open(train_dataset_file, 'rb') as f:
            target_train_dataset = pickle.load(f)
        print('{} has been loaded.'.format(train_dataset_file))
    else:
        train_dataset = None
    if os.path.isfile(valid_dataset_file):
        with open(valid_dataset_file, 'rb') as f:
            target_valid_dataset = pickle.load(f)
        print('{} has been loaded.'.format(valid_dataset_file))
    else:
        valid_dataset = None
    return target_train_dataset, target_valid_dataset

def save_checkpoint(epoch_file, model_file, opt_file, epoch, model, opt):
    for param in model.parameters():
        device = param.data.device
        break
    torch.save(model.to('cpu').state_dict(), model_file)
    torch.save(opt.state_dict(), opt_file)
    with open(epoch_file, 'wb') as f:
        pickle.dump(epoch, f)
    model.to(device)


def load_checkpoint(epoch_file, model_file, opt_file, n_epochs, model, opt):
    init_epoch = 0
    if os.path.isfile(model_file) and os.path.isfile(opt_file):
        model.load_state_dict(torch.load(model_file))
        opt.load_state_dict(torch.load(opt_file))
        print('{} has been loaded.'.format(model_file))
        print('{} has been loaded.'.format(opt_file))
        with open(epoch_file, 'rb') as f:
            init_epoch = pickle.load(f)
            last_epoch = n_epochs + init_epoch
    return init_epoch, last_epoch, model, opt


def randn_truncated(shape, a:float=-1, b:float=1):
    u = torch.rand(shape)
    a = torch.tensor([a])
    b = torch.tensor([b])
    t2 = torch.tensor([2])
    Fa = 0.5 * (1 + torch.erf(a / torch.sqrt(t2)))
    Fb = 0.5 * (1 + torch.erf(b / torch.sqrt(t2)))
    return torch.sqrt(t2) * torch.erfinv(2 * ((Fb - Fa) * u + Fa) - 1)
