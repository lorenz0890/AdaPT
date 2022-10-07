from collections import Counter

import torch
from torchvision import datasets
from torch_geometric import datasets as gm_datasets
from ogb.graphproppred import PygGraphPropPredDataset as ogb_datasets
from torchvision import transforms

from initializers.custom_init import truncated_normal_variance_scaling as tnvs
from models import QLeNet5_FP, QAlexNet_FP, QResNet_FP, ResNet_MNIST, AlexNet, LeNet5, QSimple_GCN_FP, QGCN_Graph_FP, \
    GCN_Graph, GIN_Graph, QAlexNet_MNIST_FP, AlexNet_MNIST, QResNet_MNIST_FP
from models.alexnet_imagenet import AlexNet_ImageNet
from models.alexnet_imagenet_fp import QAlexNet_ImageNet_FP
from models.gin_graph_fp import QGIN_Graph_FP
from optimizers import MSGD_FP, MAdam_FP
from utils.cv import CustomDataset, CustomGraphDataset
#import torchsample.transforms as tstf
import numpy as np

def optimizer_factory(config):
    optimizer, optim_args = None, None

    if 'OPTIMIZER' in config:
        if 'SGD' == config['OPTIMIZER']['NAME']:
            optimizer, optim_args = torch.optim.SGD, dict(lr=config.getfloat('OPTIMIZER', 'LEARNING_RATE'),
                                                          momentum=config.getfloat('OPTIMIZER', 'MOMENTUM'),
                                                          weight_decay=config.getfloat('OPTIMIZER', 'L2_DECAY'))
        elif 'Adam' == config['OPTIMIZER']['NAME']:
            optimizer, optim_args = torch.optim.Adam, dict(lr=config.getfloat('OPTIMIZER', 'LEARNING_RATE'),
                                                   betas=(0.9, 0.999),
                                                   eps=config.getfloat('OPTIMIZER', 'EPS'),
                                                   weight_decay=config.getfloat('OPTIMIZER', 'L2_DECAY'),
                                                   amsgrad=False)
        elif 'MSGD' == config['OPTIMIZER']['NAME']:
            optimizer, optim_args = MSGD_FP, dict(lr=config.getfloat('OPTIMIZER', 'LEARNING_RATE'),
                                                  momentum=config.getfloat('OPTIMIZER', 'MOMENTUM'),
                                                  weight_decay=config.getfloat('OPTIMIZER', 'L2_DECAY'))
        elif 'MAdam' == config['OPTIMIZER']['NAME']:
            optimizer, optim_args = MAdam_FP, dict(lr=config.getfloat('OPTIMIZER', 'LEARNING_RATE'),
                                                   betas=(0.9, 0.999),
                                                   eps=config.getfloat('OPTIMIZER', 'EPS'),
                                                   weight_decay=config.getfloat('OPTIMIZER', 'L2_DECAY'),
                                                   amsgrad=False)

    else:
        print("optimizer not found")
        exit(2)

    return optimizer, optim_args


def dataset_factory(config):
    train_data, test_data = None, None
    if config['MODEL']['DATASET'] == 'mnist':
        train_data = datasets.MNIST(root="./data", train=True, download=True)
        test_data = datasets.MNIST(root="./data", train=False, download=True)
    elif config['MODEL']['DATASET'] == 'places365':
        train_data = datasets.Places365(root="./data", split='train-standard', download=True, small=True)
        test_data = datasets.Places365(root="./data",  split='val', download=True, small=True)
    elif config['MODEL']['DATASET'] == 'fmnist':
        train_data = datasets.FashionMNIST(root="./data", train=True, download=True)
        test_data = datasets.FashionMNIST(root="./data", train=False, download=True)
    elif config['MODEL']['DATASET'] == 'emnist':
        train_data = datasets.EMNIST(root="./data", train=True, download=True, split='balanced')
        test_data = datasets.EMNIST(root="./data", train=False, download=True, split='balanced')
    elif config['MODEL']['DATASET'] == 'cifar10':
        train_data = datasets.CIFAR10(root="./data", train=True, download=True)
        test_data = datasets.CIFAR10(root="./data", train=False, download=True)
    elif config['MODEL']['DATASET'] == 'cifar100':
        train_data = datasets.CIFAR100(root="./data", train=True, download=True)
        test_data = datasets.CIFAR100(root="./data", train=False, download=True)
    elif config['MODEL']['DATASET'] == 'imagenet':
        train_data = datasets.ImageFolder(root="./data/imagenet/train", transform=transforms.Compose(
                                           [
                                               #transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                               #transforms.
                                           ]))
        test_data = datasets.ImageFolder(root="./data/imagenet/val", transform=transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))
    elif config['MODEL']['DATASET'] == 'cora':
        data = gm_datasets.Planetoid(root="./data", name="Cora").shuffle()
        # Simple toy dataset for testing - usually we need to split
        train_data = data  # data[:int(len(data)*0.7)]
        test_data = data
    elif config['MODEL']['DATASET'] == 'ogbg-molhiv':
        dataset = ogb_datasets(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        train_data = dataset[split_idx["train"]]
        valid_data = dataset[split_idx["valid"]]
        test_data = dataset[split_idx["test"]]
    elif config['MODEL']['DATASET'] == 'ogbg-molpcba':
        dataset = ogb_datasets(name='ogbg-molpcba')
        split_idx = dataset.get_idx_split()
        train_data = dataset[split_idx["train"]]
        valid_data = dataset[split_idx["valid"]]
        test_data = dataset[split_idx["test"]]

    else:
        print("dataset not found")
        exit(1)

    return train_data, test_data

def model_factory(config):
    model, model_args, image_data = None, None, None

    train_data, test_data = dataset_factory(config)
    optimizer, optim_args = optimizer_factory(config)

    data_args = dict(train_dataset=train_data, test_dataset=test_data,
                     batch_size=config.getint('MODEL', 'BATCH_SIZE'),
                     num_workers=config.getint('SYSTEM', 'CPUS'))

    if 'QUANTIZATION' in config:
        #print(Counter(train_data.targets), flush=True)
        #exit(1)
        #print(len(train_data.dataset) / config.getint('MODEL', 'BATCH_SIZE'), flush=True)
        model_args = dict(initializer=tnvs, initializer_arg=config.getfloat('MODEL', 'INITIALIZER_ARG'),
                          num_classes=config.getint('MODEL', 'NUM_CLASS'),
                          quant_in=config.getboolean('QUANTIZATION', 'QUANT_INPUTS'),
                          quant_out=config.getboolean('QUANTIZATION', 'QUANT_OUTPUTS'),
                          optimizer=optimizer, accumulation_steps=config.getint('MODEL', 'ACCUMULATION_STEPS'),
                          **optim_args,
                          max_lookback=config.getint('QUANTIZATION', 'MAX_LOOKBACK'),
                          push_up_strategy=config['QUANTIZATION']['PUSH_UP_STRATEGY'],
                          max_resolution=config.getint('QUANTIZATION', 'MAX_RESOLUTION'),
                          min_lookback=config.getint('QUANTIZATION', 'MIN_LOOKBACK'),
                          min_resolution=config.getint('QUANTIZATION', 'MIN_RESOLUTION'),
                          quant_grads=config.getboolean('QUANTIZATION', 'QUANT_GRADS'),
                          results_buffer=config.getint('QUANTIZATION', 'RESULTS_BUFFER'),
                          lookback_momentum=config.getfloat('QUANTIZATION', 'LOOKBACK_MOMENTUM'))
    else:
        model_args = dict(initializer=tnvs, initializer_arg=config.getfloat('MODEL', 'INITIALIZER_ARG'),
                          num_classes=config.getint('MODEL', 'NUM_CLASS'),
                          optimizer=optimizer, accumulation_steps=config.getint('MODEL', 'ACCUMULATION_STEPS'),
                          l1_decay=config.getfloat('OPTIMIZER', 'L1_DECAY'),
                          **optim_args)
    if 'GRAPH' in config:
        model_args['hidden_dim'] = config.getint('GRAPH', 'WIDTH')
        model_args['num_layers'] = config.getint('GRAPH', 'DEPTH')
        model_args['dropout'] = config.getfloat('GRAPH', 'DROPOUT')

    if 'SGD' == config['OPTIMIZER']['NAME'] or 'MSGD' == config['OPTIMIZER']['NAME']:
        model_args['patience'] = config.getint('MODEL', 'PATIENCE')
        model_args['threshold'] = config.getfloat('MODEL', 'THRESHOLD')

    if config['MODEL']['ARCHITECTURE'] == 'qlenet_fp':
        model = QLeNet5_FP(**model_args)
        image_data = CustomDataset(**data_args,
                                   train_transform=transforms.Compose(
                                       [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                   test_transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                                                                      ]))

    elif config['MODEL']['ARCHITECTURE'] == 'qalexnet_fp':
        model = QAlexNet_FP(**model_args)
        if config['MODEL']['DATASET'] == 'imagenet':
            model = QAlexNet_ImageNet_FP(**model_args)
            image_data = [None, None]
            image_data[0] = train_data
            image_data[1] = test_data

        elif 'mnist' in config['MODEL']['DATASET']:
                model = QAlexNet_MNIST_FP(**model_args)
                image_data = CustomDataset(**data_args,
                                           train_transform=transforms.Compose(
                                               [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                           test_transform=transforms.Compose(
                                               [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
        else:
            image_data = CustomDataset(**data_args,
                                       train_transform=transforms.Compose(
                                           [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                       test_transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                               (
                                                                                                   0.2023, 0.1994,
                                                                                                   0.2010)),
                                                                          ]))
    elif config['MODEL']['ARCHITECTURE'] == 'qresnet_fp':
        model = QResNet_FP(**model_args)
        image_data = CustomDataset(**data_args,
                                   train_transform=transforms.Compose(
                                       [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                   test_transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                           (0.2023, 0.1994, 0.2010)),
                                                                      ]))
        if 'mnist' in config['MODEL']['DATASET']:
            model = QResNet_MNIST_FP(**model_args)
            image_data = CustomDataset(**data_args,
                                       train_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]),
                                       test_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    elif config['MODEL']['ARCHITECTURE'] == 'qsimple_gnn_fp':
        model = QSimple_GCN_FP(**model_args)
        image_data = CustomGraphDataset(**data_args, train_transform=None,
                                   test_transform=None)
    elif config['MODEL']['ARCHITECTURE'] == 'qgcn_graph_fp':
        model_args["evaluator_name"] = config['MODEL']['DATASET']
        model = QGCN_Graph_FP(**model_args)
        image_data = CustomGraphDataset(**data_args,
                                   train_transform=None,
                                   test_transform=None)

    elif config['MODEL']['ARCHITECTURE'] == 'qgin_graph_fp':
        model_args["evaluator_name"] = config['MODEL']['DATASET']
        model = QGIN_Graph_FP(**model_args)
        image_data = CustomGraphDataset(**data_args,
                                   train_transform=None,
                                   test_transform=None)

    elif config['MODEL']['ARCHITECTURE'] == 'resnet':
        model = ResNet_MNIST(**model_args)
        image_data = CustomDataset(**data_args,
                                   train_transform=transforms.Compose(
                                       [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                   test_transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                           (0.2023, 0.1994, 0.2010)),
                                                                      ]))
        if 'mnist' in config['MODEL']['DATASET']:
            model = ResNet_MNIST(**model_args)
            image_data = CustomDataset(**data_args,
                                       train_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]),
                                       test_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    elif config['MODEL']['ARCHITECTURE'] == 'alexnet':
        model = AlexNet(**model_args)
        if config['MODEL']['DATASET'] == 'imagenet':
            model = AlexNet_ImageNet(**model_args)
            image_data = [None, None]
            image_data[0] = train_data
            image_data[1] = test_data
        elif 'mnist' in config['MODEL']['DATASET']:
            model = AlexNet_MNIST(**model_args)
            image_data = CustomDataset(**data_args,
                                       train_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]),
                                       test_transform=transforms.Compose(
                                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
        else:
            image_data = CustomDataset(**data_args,
                                       train_transform=transforms.Compose(
                                           [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                       test_transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                               (
                                                                                                   0.2023, 0.1994,
                                                                                                   0.2010)),
                                                                      ]))

    elif config['MODEL']['ARCHITECTURE'] == 'lenet':
        model = LeNet5(**model_args)
        image_data = CustomDataset(**data_args,
                                   train_transform=transforms.Compose(
                                       [transforms.Resize((32, 32)), transforms.ToTensor()]),
                                   test_transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()
                                                                      ]))
    elif config['MODEL']['ARCHITECTURE'] == 'gcn_graph':
        model_args["evaluator_name"] = config['MODEL']['DATASET']
        model = GCN_Graph(**model_args)
        image_data = CustomGraphDataset(**data_args,
                                   train_transform=None,
                                   test_transform=None)
    elif config['MODEL']['ARCHITECTURE'] == 'gin_graph':
        model_args["evaluator_name"] = config['MODEL']['DATASET']
        model = GIN_Graph(**model_args)
        image_data = CustomGraphDataset(**data_args,
                                   train_transform=None,
                                   test_transform=None)

    else:
        print("model not found")
        exit(3)

    return model, model_args, image_data
