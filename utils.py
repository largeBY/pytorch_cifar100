""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16_bn':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13_bn':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11_bn':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19_bn':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19()
    elif args.net == 'cascadedfcn':
        from models.cascadedfcn import cascaded_fcn
        net = cascaded_fcn()
    elif args.net == 'voc':
        from models.vocnet import vocnet
        net = vocnet()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def segmentation_custom_collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])
        target = sample[1]

        if target.ndim == 3:
            target = target.squeeze(0)  # 移除不必要的维度

        targets.append(target)

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    return images, targets


def detection_custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def get_training_dataloader(args, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        training = torchvision.datasets.CIFAR100(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        training_loader = DataLoader(
            training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size)
    elif args.dataset == 'VOCSegmentation':
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整到适当的尺寸
            transforms.RandomCrop(224),  # 可以根据需要修改尺寸
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        target_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整到适当的尺寸
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])

        training = torchvision.datasets.VOCSegmentation(root='./data',
                                                        year="2012",
                                                        download=True,
                                                        image_set="train",
                                                        transform=transform_train,
                                                        target_transform=target_transform)
        training_loader = DataLoader(
            training,
            shuffle=shuffle,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=segmentation_custom_collate_fn)
    elif args.dataset == 'VOCDetection':
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # 调整到适当的尺寸
            transforms.RandomCrop(224),  # 可以根据需要修改尺寸
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        def target_transform(target):
            objects = target['annotation']['object']

            if not isinstance(objects, list):
                objects = [objects]  # 如果只有一个对象，转换为列表

            boxes = []
            labels = []

            for obj in objects:
                bndbox = obj['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASSES.index(obj['name']))  # 假设 CLASSES 是类别名的列表

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            return {'boxes': boxes, 'labels': labels}

        training = torchvision.datasets.VOCDetection(root='./data',
                                                     year="2012",
                                                     download=True,
                                                     image_set="train",
                                                     transform=transform_train,
                                                     target_transform=target_transform)
        training_loader = DataLoader(
            training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=detection_custom_collate_fn)
    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()

    # for images, labels in training_loader:
    #     print(f'Batch size: {images.size(0)}, Image size: {images.size()}')
    #     break  # 只打印第一个批次的数据

    return training_loader


def get_test_dataloader(args, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test = torchvision.datasets.CIFAR100(root='./data',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
        test_loader = DataLoader(
            test,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size)
    elif args.dataset == 'VOCSegmentation':
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        target_transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        test = torchvision.datasets.VOCSegmentation(root='./data',
                                                    year="2012",
                                                    download=True,
                                                    image_set="val",
                                                    transform=transform_test,
                                                    target_transform=target_transform_test)
        test_loader = DataLoader(
            test, shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=segmentation_custom_collate_fn)
    elif args.dataset == 'VOCDetection':
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        target_transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        test = torchvision.datasets.VOCDetection(root='./data',
                                                 year="2012",
                                                 download=True,
                                                 image_set="val",
                                                 transform=transform_test,
                                                 target_transform=target_transform_test)
        test_loader = DataLoader(
            test, shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=detection_custom_collate_fn)
    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()

    return test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]