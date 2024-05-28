# import os
#
# import numpy as np
# import torch
from torch.utils.data.dataset import Subset
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# from utils_.utils import set_random_seed
# from cutpast_transformation import *
# from PIL import Image
# from glob import glob
# import random

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
from custom_datasets import *
from torch.utils.data import ConcatDataset

import medmnist
from medmnist import INFO, Evaluator

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'

CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class
TUMOR_BRAIN_SUPERCLASS = list(range(2))
MNIST_SUPERCLASS = list(range(10))
SVHN_SUPERCLASS = list(range(10))
FashionMNIST_SUPERCLASS = list(range(10))
MVTecAD_SUPERCLASS = list(range(2))
HEAD_CT_SUPERCLASS = list(range(2))
ART_BENCH_SUPERCLASS = list(range(10))
MVTEC_HV_SUPERCLASS = list(range(2))
breastmnist_SUPERCLASS = list(range(2))
CIFAR100_SUPERCLASS = list(range(20))
UCSD_SUPERCLASS = list(range(2))
CIFAR10_CORRUPTION_SUPERCLASS = list(range(10))
MNIST_CORRUPTION_SUPERCLASS = list(range(10))
CIFAR10_VER_CIFAR100_SUPERCLASS = list(range(2))
DTD_SUPERCLASS = list(range(46))
WBC_SUPERCLASS = list(range(2))
DIOR_SUPERCLASS = list(range(19))
ISIC2018_SUPERCLASS = list(range(2))


def sparse2coarse(targets):
    coarse_labels = np.array(
        [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3,
         14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5,
         10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
         12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
         13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
         17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12,
         1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
         15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
         19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13, ])
    return coarse_labels[targets]


CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut',
               'hazelnut', 'screw', 'carpet', 'leather', 'cable']


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def mvtecad_dataset(P, category, root="./mvtec_anomaly_detection", image_size=(224, 224, 3)):
    # image_size = (224, 224, 3)
    n_classes = 2
    categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule',
                  'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((image_size[0], image_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.ToTensor(),
    ])

    test_ds_mvtech = MVTecDataset(root=root, train=False, category=categories[category], transform=test_transform,
                                  count=-1)
    train_ds_mvtech_normal = MVTecDataset(root=root, train=True, category=categories[category],
                                          transform=train_transform, count=P.main_count)

    print("test_ds_mvtech shapes: ", test_ds_mvtech[0][0].shape)
    print("train_ds_mvtech_normal shapes: ", train_ds_mvtech_normal[0][0].shape)

    return train_ds_mvtech_normal, test_ds_mvtech, image_size, n_classes


def get_exposure_dataloader(P, batch_size=64, image_size=(224, 224, 3),
                            base_path='./tiny-imagenet-200', fake_root="./fake_mvtecad",
                            root="./mvtec_anomaly_detection", count=-1, cls_list=None, labels=None):
    categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule',
                  'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
    if P.dataset == 'high-variational-brain-tumor' or P.dataset == 'head-ct' or P.dataset == 'breastmnist' or P.dataset == 'mnist' or P.dataset == 'fashion-mnist' or P.dataset == 'Tomor_Detection':
        tiny_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.AutoAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif P.dataset == 'cifar10-versus-100' or P.dataset == 'cifar100-versus-10':
        tiny_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomChoice(
                [transforms.RandomApply(
                    [transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                    transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)],
                                           p=0.6),
                    transforms.RandomApply([transforms.AutoAugment()], p=0.9), ]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        tiny_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.AutoAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    fake_count = int(P.fake_data_percent * count)
    tiny_count = int((1 - (P.fake_data_percent + P.cutpast_data_percent)) * count)
    cutpast_count = int(P.cutpast_data_percent * count)
    if (fake_count + tiny_count + cutpast_count) != count:
        tiny_count += (count - (cutpast_count + fake_count + tiny_count))
    print("fake_count, tiny_count, cutpast_count", fake_count, tiny_count, cutpast_count)
    if P.dataset == "MVTecAD":
        fake_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_transform_cutpasted = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
        ])
        imagenet_exposure = ImageNetExposure(root=base_path, count=tiny_count, transform=tiny_transform)
        train_ds_mvtech_fake = FakeMVTecDataset(root=fake_root, train=True, category=categories[P.one_class_idx],
                                                transform=fake_transform, count=fake_count)
        train_ds_mvtech_cutpasted = MVTecDataset_Cutpasted(root=root, train=True, category=categories[P.one_class_idx],
                                                           transform=train_transform_cutpasted, count=cutpast_count)
        print("number of fake data:", len(train_ds_mvtech_fake), 'shape:', train_ds_mvtech_fake[0][0].shape)
        print("number of tiny data:", len(imagenet_exposure), 'shape:', imagenet_exposure[0][0].shape)
        print("number of cutpasted data:", len(train_ds_mvtech_cutpasted), 'shape:',
              train_ds_mvtech_cutpasted[0][0].shape)
        exposureset = torch.utils.data.ConcatDataset(
            [train_ds_mvtech_fake, imagenet_exposure, train_ds_mvtech_cutpasted])

        print("number of exposure:", len(exposureset))
        train_loader = DataLoader(exposureset, batch_size=batch_size, shuffle=True)
    elif P.dataset == "mvtec-high-var":
        fake_root = './fake_mvtecad'
        fake_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_transform_cutpasted = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
        ])
        imagenet_exposure = ImageNetExposure(root=base_path, count=tiny_count, transform=tiny_transform)
        fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
        if sum(fc) != fake_count:
            fc[0] += abs(fake_count - sum(fc))
        print("fake couns:", fc)
        fcp = [int(cutpast_count / len(cls_list)) for i in range(len(cls_list))]
        if sum(fcp) != cutpast_count:
            fcp[0] += abs(cutpast_count - sum(fcp))
        print("cutpast couns:", fcp)
        train_ds_mvtech_fake = []
        train_ds_mvtech_cutpasted = []
        for idx, i in enumerate(cls_list):
            train_ds_mvtech_fake.append(
                FakeMVTecDataset(root=fake_root, train=True, category=categories[i], transform=fake_transform,
                                 count=fc[idx]))
            train_ds_mvtech_cutpasted.append(MVTecDataset_Cutpasted(root=root, train=True, category=categories[i],
                                                                    transform=train_transform_cutpasted,
                                                                    count=fcp[idx]))
        train_ds_mvtech_cutpasted = ConcatDataset(train_ds_mvtech_cutpasted)
        train_ds_mvtech_fake = ConcatDataset(train_ds_mvtech_fake)

        exposureset = torch.utils.data.ConcatDataset(
            [train_ds_mvtech_fake, imagenet_exposure, train_ds_mvtech_cutpasted])
        if len(train_ds_mvtech_fake) > 0:
            print("number of fake data:", len(train_ds_mvtech_fake), "shape:", train_ds_mvtech_fake[0][0].shape)
        if len(train_ds_mvtech_cutpasted) > 0:
            print("number of cutpast data:", len(train_ds_mvtech_cutpasted), 'shape:',
                  train_ds_mvtech_cutpasted[0][0].shape)
        print("number of tiny data:", len(imagenet_exposure), 'shape:', imagenet_exposure[0][0].shape)
        print("number of exposure:", len(exposureset))
        train_loader = DataLoader(exposureset, batch_size=batch_size, shuffle=True)
    else:
        if P.dataset == 'breastmnist' or P.dataset == 'mnist' or P.dataset == 'fashion-mnist':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation((90, 270)),
                CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])
        elif P.dataset == 'head-ct':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation((90, 270)),
                High_CutPasteUnion(),
            ])
        elif P.dataset == 'Tomor_Detection':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation((90, 270)),
                CutPasteNormal(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])
        elif P.dataset == 'dtd' or P.dataset == 'cub-birds':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                High_CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])
        elif P.dataset == 'WBC':
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToPILImage(),
                High_CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])
        else:
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])
        if P.dataset == 'WBC' or P.dataset == 'cub-birds':
            cutpast_train_set, _, _, _ = get_dataset(P, dataset=P.dataset, download=True, image_size=image_size,
                                                     train_transform_cutpasted=train_transform_cutpasted,
                                                     labels=cls_list)
        else:
            cutpast_train_set, _, _, _ = get_dataset(P, dataset=P.dataset, download=True, image_size=image_size,
                                                     train_transform_cutpasted=train_transform_cutpasted)
        print("len(cutpast_train_set) before set_count: ", len(cutpast_train_set))
        if P.dataset == 'cub-birds' or P.dataset == 'head-ct' or P.dataset == 'mvtec-high-var' or P.dataset == 'ucsd' or P.dataset == 'WBC' or P.dataset == 'cifar100-versus-10' or P.dataset == 'cifar10-versus-100':
            cutpast_train_set = set_dataset_count(cutpast_train_set, count=cutpast_count)
        else:
            print("cls_list(normal class)", cls_list)
            cutpast_train_set = get_subclass_dataset(cutpast_train_set, classes=cls_list, count=cutpast_count)
        print("len(cutpast_train_set) after set_count: ", len(cutpast_train_set))

        imagenet_exposure = ImageNetExposure(root=base_path, count=tiny_count, transform=tiny_transform)
        if P.dataset == 'cub-birds' or P.dataset == 'STL-10':
            image_path = glob('./one_class_train/*/*')[:tiny_count]
            imagenet_exposure = ImageNet30_Dataset(image_path=image_path, labels=[1] * len(image_path),
                                                   transform=tiny_transform)

        if P.dataset == "cifar10":
            """
            tiny_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomChoice(
                    [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                    transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),]),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            
            # tiny1_cnt=int(tiny_count/2) 
            # tiny2_cnt = tiny_count - tiny1_cnt
            tiny1_cnt = 97500
            imagenet_exposure1 = ImageNetExposure(root=base_path, count=tiny1_cnt, transform=tiny_transform)
            imagenet_exposure2 = datasets.ImageFolder('./one_class_train', transform=tiny_transform)
            tiny2_cnt = 39000 
            '''
            unique_numbers = [] 
            while len(unique_numbers) < tiny2_cnt:
                number = random.randint(0, len(imagenet_exposure2)-1)
                if number not in unique_numbers:
                    unique_numbers.append(number)
            imagenet_exposure2 = Subset(imagenet_exposure2, unique_numbers)
            '''
            imagenet_exposure = torch.utils.data.ConcatDataset([imagenet_exposure1, imagenet_exposure2])
            """
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './CIFAR10-Fake/'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_cifar10_fake = FakeCIFAR10(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_cifar10_fake) > 0:
                print("number of fake data:", len(train_ds_cifar10_fake), "shape:", train_ds_cifar10_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_cifar10_fake, imagenet_exposure])
        elif P.dataset == "cifar10-versus-100":
            cls_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './CIFAR10-Fake/'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_cifar10_fake = FakeCIFAR10(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_cifar10_fake) > 0:
                print("number of fake data:", len(train_ds_cifar10_fake), "shape:", train_ds_cifar10_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_cifar10_fake, imagenet_exposure])

        elif P.dataset == "cifar100-versus-10":
            cls_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_cifar100_fake = FakeCIFAR100(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_cifar100_fake) > 0:
                print("number of fake data:", len(train_ds_cifar100_fake), "shape:", train_ds_cifar100_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_cifar100_fake, imagenet_exposure])

        elif P.dataset == "cifar100":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_cifar100_fake = FakeCIFAR100(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_cifar100_fake) > 0:
                print("number of fake data:", len(train_ds_cifar100_fake), "shape:", train_ds_cifar100_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_cifar100_fake, imagenet_exposure])

        elif P.dataset == "dtd":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_dtd_fake = FakeDTD(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_dtd_fake) > 0:
                print("number of fake data:", len(train_ds_dtd_fake), "shape:", train_ds_dtd_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_dtd_fake, imagenet_exposure])

        elif P.dataset == "svhn-10":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fake_root = './SVHN-Fake/'
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            train_ds_svhn_fake = Fake_SVHN_Dataset(root=fake_root, category=cls_list, transform=fake_transform,
                                                   count=fc)
            if len(train_ds_svhn_fake) > 0:
                print("number of fake data:", len(train_ds_svhn_fake), "shape:", train_ds_svhn_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_svhn_fake, imagenet_exposure])

        elif P.dataset == "mnist":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            fake_root = './MNIST-Fake/'
            train_ds_mnist_fake = FakeMNIST(root=fake_root, category=cls_list, transform=fake_transform, count=fc)
            if len(train_ds_mnist_fake) > 0:
                print("number of fake data:", len(train_ds_mnist_fake), "shape:", train_ds_mnist_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_mnist_fake, imagenet_exposure])
        elif P.dataset == "fashion-mnist":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            fc = [int(fake_count / len(cls_list)) for i in range(len(cls_list))]
            if sum(fc) != fake_count:
                fc[0] += abs(fake_count - sum(fc))
            fake_root = './Fashion-Fake/'
            train_ds_fmnist_fake = FakeFashionDataset(root=fake_root, category=cls_list, transform=fake_transform,
                                                      count=fc)
            if len(train_ds_fmnist_fake) > 0:
                print("number of fake data:", len(train_ds_fmnist_fake), "shape:", train_ds_fmnist_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_fmnist_fake, imagenet_exposure])
        elif P.dataset == "Tomor_Detection":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            train_ds_tumor_detection_fake = AdaptiveExposure(root='./AdaptiveExposure/', transform=fake_transform,
                                                             count=fake_count)
            if len(train_ds_tumor_detection_fake) > 0:
                print("number of fake data:", len(train_ds_tumor_detection_fake), "shape:",
                      train_ds_tumor_detection_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset(
                [cutpast_train_set, train_ds_tumor_detection_fake, imagenet_exposure])

        elif P.dataset == "head-ct":
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            train_ds_head_ct_fake = HEAD_CT_FAKE(transform=fake_transform, count=fake_count)
            if len(train_ds_head_ct_fake) > 0:
                print("number of fake data:", len(train_ds_head_ct_fake), "shape:", train_ds_head_ct_fake[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, train_ds_head_ct_fake, imagenet_exposure])
        elif P.dataset == 'WBC':
            fake_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            wbc_fake_dataset = FakeWBC(count=fake_count, transform=fake_transform)
            if len(wbc_fake_dataset) > 0:
                print("number of fake data:", len(wbc_fake_dataset), "shape:", wbc_fake_dataset[0][0].shape)
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, wbc_fake_dataset, imagenet_exposure])
        else:
            exposureset = torch.utils.data.ConcatDataset([cutpast_train_set, imagenet_exposure])

        if len(cutpast_train_set) > 0:
            print("number of cutpast data:", len(cutpast_train_set), 'shape:', cutpast_train_set[0][0].shape)
        print("number of tiny data:", len(imagenet_exposure), 'shape:', imagenet_exposure[0][0].shape)
        print("number of exposure:", len(exposureset))
        train_loader = DataLoader(exposureset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_breastmnist_test(normal_class_indx, path, transform):
    data_flag = 'breastmnist'
    BATCH_SIZE = 64
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    test_dataset = DataClass(split='test', transform=transform, download=download)
    test_dataset.labels = test_dataset.labels.squeeze()
    return test_dataset


def get_breastmnist_train(anomaly_class_indx, path, transform):
    data_flag = 'breastmnist'
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=transform, download=download)
    train_dataset.labels = train_dataset.labels.squeeze()
    return train_dataset


def get_dataset(P, dataset, test_only=False, image_size=(32, 32, 3), download=False, eval=False,
                train_transform_cutpasted=None, labels=None):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        # image_size = (32, 32, 3)
        n_classes = 10
        if train_transform_cutpasted:
            train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform_cutpasted)
        else:
            train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'ArtBench':
        n_classes = 10
        if train_transform_cutpasted:
            train_set = ArtBench10(root=DATA_PATH, train=True, download=True, transform=train_transform_cutpasted)
        else:
            train_set = ArtBench10(root=DATA_PATH, train=True, download=True, transform=train_transform)
        test_set = ArtBench10(root=DATA_PATH, train=False, download=True, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'dior':
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        _transform = train_transform_cutpasted if train_transform_cutpasted else transform
        train_set = DIOR(DATA_PATH, train=True, download=download, transform=_transform)
        n_classes = len(train_set.classes)
        test_set = DIOR(DATA_PATH, train=False, download=download, transform=transform)
    elif dataset == 'ucsd':
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = UCSDDataset(root="./", is_normal=True, transform=train_transform_cutpasted)
        else:
            train_set = UCSDDataset(root="./", is_normal=True, transform=transform)
        test_set = UCSDDataset(root="./", is_normal=False, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'Tomor_Detection':
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = TumorDetection(transform=train_transform_cutpasted, train=True)
        else:
            train_set = TumorDetection(transform=transform, train=True)
        test_set = TumorDetection(transform=transform, train=False)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar10-versus-100':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform_cutpasted)
        else:
            train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)

        for i in range(len(train_set)):
            train_set.targets[i] = 0

        anomaly_testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        for i in range(len(anomaly_testset)):
            anomaly_testset.targets[i] = 1
        normal_testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset])
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar100-versus-10':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform_cutpasted)
        else:
            train_set = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)

        for i in range(len(train_set)):
            train_set.targets[i] = 0

        anomaly_testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        for i in range(len(anomaly_testset)):
            anomaly_testset.targets[i] = 1
        normal_testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset])
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'head-ct':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        import pandas as pd
        labels_df = pd.read_csv('./head-ct/labels.csv')
        labels_ = np.array(labels_df[' hemorrhage'].tolist())
        images = np.array(sorted(glob('./head-ct/head_ct/head_ct/*.png')))
        np.random.seed(1225)
        indicies = np.random.permutation(100)
        train_true_idx, test_true_idx = indicies[:75] + 100, indicies[75:] + 100
        train_false_idx, test_false_idx = indicies[:75], indicies[75:]
        train_idx, test_idx = train_true_idx, np.concatenate((test_true_idx, test_false_idx, train_false_idx))

        train_image, train_label = images[train_idx], labels_[train_idx]
        test_image, test_label = images[test_idx], labels_[test_idx]

        print("train_image.shape, test_image.shape: ", train_image.shape, test_image.shape)
        print("train_label.shape, test_label.shape: ", train_label.shape, test_label.shape)
        if train_transform_cutpasted:
            train_set = HEAD_CT_DATASET(image_path=train_image, labels=train_label, transform=train_transform_cutpasted)
        else:
            train_set = HEAD_CT_DATASET(image_path=train_image, labels=train_label, transform=train_transform)
        test_set = HEAD_CT_DATASET(image_path=test_image, labels=test_label, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'breastmnist':
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_set = get_breastmnist_test(normal_class_indx=P.one_class_idx, path='./data/', transform=transform)
        if train_transform_cutpasted:
            train_set = get_breastmnist_train(anomaly_class_indx=P.one_class_idx, path='./data/',
                                              transform=train_transform_cutpasted)
        else:
            train_set = get_breastmnist_train(anomaly_class_indx=P.one_class_idx, path='./data/', transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'WBC':
        n_classes = 2
        data_path_good_train = "./CELL_MIR"
        orig_transform = transforms.Compose([
            transforms.Resize([image_size[0], image_size[1]]), transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset_train_good = ImageFolder(root=data_path_good_train, transform=orig_transform)
        CELL_train_loader = torch.utils.data.DataLoader(
            dataset_train_good,
            batch_size=4,
            shuffle=False,
        )
        training_cell = []
        for x in CELL_train_loader:
            training_cell.append(x[0])
        training_cell = torch.cat(training_cell)

        import pandas as pd
        df = pd.read_csv("./segmentation_WBC/Class Labels of Dataset 1.csv")
        rows = []
        for row in df['class label']:
            rows.append(row)
        images1 = []
        images2 = []
        images3 = []
        images4 = []
        index = [0, 0, 0, 0, 0, 0]
        for i in range(300):
            if rows[i] == 1: images1.append(training_cell[i])
            if rows[i] == 2: images2.append(training_cell[i])
            if rows[i] == 3: images3.append(training_cell[i])
            if rows[i] == 4: images4.append(training_cell[i])

        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        images3 = torch.stack(images3)
        images4 = torch.stack(images4)

        np.random.shuffle(images1.numpy())
        np.random.shuffle(images2.numpy())
        np.random.shuffle(images3.numpy())
        np.random.shuffle(images4.numpy())

        Normal_data = None
        data = [images1, images2, images3, images4]

        normal_train_data = []
        normal_train_label = []

        normal_test_data = []
        normal_test_label = []

        for i in labels:
            if normal_train_data:
                normal_train_data = torch.cat([normal_train_data, data[i][:int(data[i].shape[0] * 0.8)]])
                normal_test_data = torch.cat([normal_test_data, data[i][int(data[i].shape[0] * 0.8):]])
            else:
                normal_train_data = data[i][:int(data[i].shape[0] * 0.8)]
                normal_test_data = data[i][int(
                    data[i].shape[0] * 0.8):]  # normal_test_data += data[i][int(images1.shape[0]*0.8):]

            normal_train_label.append([0] * int(data[i].shape[0] * 0.8))
            normal_test_label.append([0] * (data[i].shape[0] - int(data[i].shape[0] * 0.8)))

        normal_train_label = np.concatenate(normal_train_label)
        normal_test_label = np.concatenate(normal_test_label)
        # normal_train_data.shape, len(normal_train_label), normal_test_data.shape, len(normal_test_label)

        test_set_ = []
        test_set_.append(normal_test_data)
        for i in range(4):
            if i not in labels:
                test_set_.append(data[i])
        test_set__t = torch.cat(test_set_)
        # len(test_set__t), test_set__t.shape

        test_label = []
        test_label.append(normal_test_label)
        for i in range(4):
            if i not in labels:
                test_label.append([1] * (data[i].shape[0]))
        test_label = np.concatenate(test_label)
        # len(test_label), test_label.shape, test_label

        orig_transform_224 = transforms.Compose([
            transforms.Resize([image_size[0], image_size[1]])
        ])
        _transform_224 = transforms.Compose([
            transforms.Resize([image_size[0], image_size[1]]), transforms.RandomHorizontalFlip()
        ])

        test_set = MyDataset_Binary(test_set__t, torch.tensor(test_label), transform=orig_transform_224)
        if train_transform_cutpasted:
            train_set = MyDataset_Binary(normal_train_data, normal_train_label, transform=train_transform_cutpasted)
        else:
            train_set = MyDataset_Binary(normal_train_data, normal_train_label, transform=orig_transform_224)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        print("len(train_set), len(test_set): ", len(train_set), len(test_set))

        '''
        n_classes = 2        
        orig_transform_224 = transforms.Compose([
            transforms.Resize([image_size[0], image_size[1]]),
            transforms.ToTensor(),
        ])
        _transform_224 = transforms.Compose([
            transforms.Resize([image_size[0], image_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_set = WBC_DATASET(train=False, transform=orig_transform_224, normal_set=labels)
        if train_transform_cutpasted:
            train_set = WBC_DATASET(train=True, transform=train_transform_cutpasted, normal_set=labels)
        else:
            train_set = WBC_DATASET(train=True, transform=_transform_224, normal_set=labels)
        print("train_set shapes: ", train_set[0][0].shape, len(train_set))
        print("test_set shapes: ", test_set[0][0].shape, len(test_set))
        '''

    elif dataset == 'mvtec-high-var':
        n_classes = 2
        train_dataset = []
        test_dataset = []
        root = "./mvtec_anomaly_detection"
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        for class_idx in labels:
            if train_transform_cutpasted:
                train_dataset.append(MVTecDataset_Cutpasted(root=root, train=True, category=CLASS_NAMES[class_idx],
                                                            transform=train_transform_cutpasted, count=-1))
            else:
                train_dataset.append(
                    MVTecDataset(root=root, train=True, category=CLASS_NAMES[class_idx], transform=train_transform,
                                 count=-1))
            test_dataset.append(
                MVTecDataset(root=root, train=False, category=CLASS_NAMES[class_idx], transform=test_transform,
                             count=-1))

        train_set = ConcatDataset(train_dataset)
        test_set = ConcatDataset(test_dataset)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

        print("len(test_dataset), len(train_dataset)", len(test_set), len(train_set))
    ####################################################3
    elif dataset == 'chest':
        n_classes = 2
        train_dataset = []
        test_dataset = []
        root = "./mvtec_anomaly_detection"
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        test_set = chest(transform=transform, train=True)
        train_set = chest(transform=transform, train=False)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

        ####################################################################333333
    elif dataset == 'mvtec-high-var-corruption':
        n_classes = 2
        train_dataset = []
        test_dataset = []
        root = "./mvtec_anomaly_detection"
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        def gaussian_noise(image, mean=P.noise_mean, std=P.noise_std, noise_scale=P.noise_scale):
            image = image + (torch.randn(image.size()) * std + mean) * noise_scale
            return image

        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Lambda(gaussian_noise),
        ])
        for class_idx in labels:
            if train_transform_cutpasted:
                train_dataset.append(MVTecDataset_Cutpasted(root=root, train=True, category=CLASS_NAMES[class_idx],
                                                            transform=train_transform_cutpasted, count=-1))
            else:
                train_dataset.append(
                    MVTecDataset(root=root, train=True, category=CLASS_NAMES[class_idx], transform=train_transform,
                                 count=-1))
            test_dataset.append(
                MVTecDataset(root=root, train=False, category=CLASS_NAMES[class_idx], transform=test_transform,
                             count=-1))

        train_set = ConcatDataset(train_dataset)
        test_set = ConcatDataset(test_dataset)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

        print("len(test_dataset), len(train_dataset)", len(test_set), len(train_set))
    elif dataset == 'fashion-mnist':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download,
                                              transform=train_transform_cutpasted)
        else:
            train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.FashionMNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'dtd':
        # image_size = (32, 32, 3)
        n_classes = 10
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.DTD('./data', split="train", download=True, transform=train_transform_cutpasted)
        else:
            train_set = datasets.DTD('./data', split="train", download=True, transform=train_transform)
        test_set = datasets.DTD('./data', split="test", download=True, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'cifar100':
        # image_size = (32, 32, 3)
        n_classes = 100
        if train_transform_cutpasted:
            train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform_cutpasted)
        else:
            train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
        test_set.targets = sparse2coarse(test_set.targets)
        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'mnist':
        # image_size = (32, 32, 1)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform_cutpasted)
        else:
            train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar10-corruption':
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar100-corruption':
        n_classes = 100
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_label='CIFAR-100-C/labels.npy',
                                    cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=transform)

        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'mnist-corruption':
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_set = MNIST_CORRUPTION(root_dir=P.mnist_corruption_folder, corruption_type=P.mnist_corruption_type,
                                    transform=transform, train=False)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'svhn-10':
        # image_size = (32, 32, 3)
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform_cutpasted)
        else:
            train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'STL-10':
        # image_size = (32, 32, 3)
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_set = datasets.STL10(DATA_PATH, split='train', download=download, transform=train_transform_cutpasted)
        else:
            train_set = datasets.STL10(DATA_PATH, split='train', download=download, transform=transform)
        test_set = datasets.STL10(DATA_PATH, split='test', download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)


    elif dataset == 'svhn-10-corruption':

        def gaussian_noise(image, mean=P.noise_mean, std=P.noise_std, noise_scale=P.noise_scale):
            image = image + (torch.randn(image.size()) * std + mean) * noise_scale
            return image

        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Lambda(gaussian_noise)
        ])

        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'high-variational-brain-tumor':
        if eval:
            head_ct_cnt = -1
            tumor_detection_cnt = None
            brain_mri_cnt = -1
        else:
            head_ct_cnt = 2000
            tumor_detection_cnt = 2000
            brain_mri_cnt = 2000

        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            # transforms.CenterCrop(224),
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            # transforms.CenterCrop(224),
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        import pandas as pd
        labels_df = pd.read_csv('./head-ct/labels.csv')
        labels_ = np.array(labels_df[' hemorrhage'].tolist())
        images = np.array(sorted(glob('./head-ct/head_ct/head_ct/*.png')))
        np.random.seed(1225)
        indicies = np.random.permutation(100)
        train_true_idx, test_true_idx = indicies[:75] + 100, indicies[75:] + 100
        train_false_idx, test_false_idx = indicies[:75], indicies[75:]
        train_idx, test_idx = train_true_idx, np.concatenate((test_true_idx, test_false_idx, train_false_idx))

        train_image, train_label = images[train_idx], labels_[train_idx]
        test_image, test_label = images[test_idx], labels_[test_idx]

        print("train_image.shape, test_image.shape: ", train_image.shape, test_image.shape)
        print("train_label.shape, test_label.shape: ", train_label.shape, test_label.shape)
        if train_transform_cutpasted:
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomApply(torch.nn.ModuleList([
                    transforms.RandomRotation((90, 270)),
                ]), p=0.3),
                High_CutPasteUnion(),
            ])
            head_ct_train_set = HEAD_CT_DATASET(image_path=list(train_image), labels=list(train_label),
                                                transform=train_transform_cutpasted, count=head_ct_cnt)
        else:
            head_ct_train_set = HEAD_CT_DATASET(image_path=list(train_image), labels=list(train_label),
                                                transform=train_transform, count=head_ct_cnt)
        head_ct_test_set = HEAD_CT_DATASET(image_path=test_image, labels=test_label, transform=test_transform)

        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                High_CutPasteUnion(),
            ])
            tumor_detc_train_set = TumorDetection(transform=train_transform_cutpasted, train=True,
                                                  count=tumor_detection_cnt)
        else:
            tumor_detc_train_set = TumorDetection(transform=transform, train=True, count=tumor_detection_cnt)
        tumor_detc_test_set = TumorDetection(transform=transform, train=False)

        normal_files = sorted(glob('./brain_tumor_dataset/no/*'))
        test_normal = normal_files[:25]
        train_normal = normal_files[25:]
        abnormal_files = sorted(glob('./brain_tumor_dataset/yes/*'))

        train_path = train_normal
        train_label = [0] * len(train_normal)
        test_path = test_normal + abnormal_files
        test_label = [0] * len(test_normal) + [1] * len(abnormal_files)

        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                High_CutPasteUnion(),
            ])
            BrainMRI_train_set = BrainMRI(image_path=train_path, labels=train_label,
                                          transform=train_transform_cutpasted, count=brain_mri_cnt)
        else:
            BrainMRI_train_set = BrainMRI(image_path=train_path, labels=train_label, transform=transform,
                                          count=brain_mri_cnt)

        BrainMRI_test_set = BrainMRI(image_path=test_path, labels=test_label, transform=transform)

        train_set = torch.utils.data.ConcatDataset([BrainMRI_train_set, tumor_detc_train_set, head_ct_train_set])
        test_set = torch.utils.data.ConcatDataset([BrainMRI_test_set, tumor_detc_test_set, head_ct_test_set])

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        print("len(test_set), len(train_set): ", len(test_set), len(train_set))
    elif dataset == 'ISIC2018':
        n_classes = 2
        train_path = glob('./ISIC_DATASET/dataset/train/NORMAL/*')
        train_label = [0] * len(train_path)

        test_anomaly_path = glob('./ISIC_DATASET/dataset/test/ABNORMAL/*')
        test_anomaly_label = [1] * len(test_anomaly_path)
        test_normal_path = glob('./ISIC_DATASET/dataset/test/NORMAL/*')
        test_normal_label = [0] * len(test_normal_path)

        test_label = test_anomaly_label + test_normal_label
        test_path = test_anomaly_path + test_normal_path

        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        if train_transform_cutpasted:
            '''
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                CutPasteScar(),
                CutPasteScar(),
                CutPasteUnion,
                CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
            ])
            '''
            train_transform_cutpasted = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                CutPasteUnion(),
                CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
            ])

            train_set = ISIC2018(image_path=train_path, labels=train_label, transform=train_transform_cutpasted)
        else:
            train_set = ISIC2018(image_path=train_path, labels=train_label, transform=transform)

        test_set = ISIC2018(image_path=test_path, labels=test_label, transform=transform)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        print("len(test_set), len(train_set): ", len(test_set), len(train_set))
    elif dataset == 'cifar100-versus-other-eval':
        n_classes = 2
        cifar_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        if P.outlier_dataset == 'mnist' or P.outlier_dataset == 'fashion-mnist':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])

        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=cifar_transform)
        train_set.targets = sparse2coarse(train_set.targets)
        # for i in range(len(train_set)):
        #    train_set.targets[i] = 0

        if P.outlier_dataset == 'svhn':
            anomaly_testset = datasets.SVHN('./data', split='test', download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.labels[i] = 1
        elif P.outlier_dataset == 'mnist':
            anomaly_testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'fashion-mnist':
            anomaly_testset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'imagenet30':
            n_classes = 2
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            image_path = glob('./one_class_test/*/*/*')
            anomaly_testset = ImageNet30_Dataset(image_path=image_path, labels=[1] * len(image_path),
                                                 transform=transform)

        normal_testset = datasets.CIFAR100('./data', train=False, download=True, transform=cifar_transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset])
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)


    elif dataset == 'cifar10-versus-other-eval':
        n_classes = 2
        cifar_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        if P.outlier_dataset == 'mnist' or P.outlier_dataset == 'fashion-mnist':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)

        if P.outlier_dataset == 'svhn':
            anomaly_testset = datasets.SVHN('./data', split='test', download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.labels[i] = 1
        elif P.outlier_dataset == 'mnist':
            anomaly_testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'fashion-mnist':
            anomaly_testset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'imagenet30':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            image_path = glob('./one_class_test/*/*/*')
            anomaly_testset = ImageNet30_Dataset(image_path=image_path, labels=[1] * len(image_path),
                                                 transform=transform)

        normal_testset = datasets.CIFAR10('./data', train=False, download=True, transform=cifar_transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0

        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset])
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'stanford-cars':
        import pandas as pd
        from scipy.io import loadmat
        n_classes = 20
        cars_train_annos = loadmat('./stanford_cars/devkit/cars_train_annos.mat')
        frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
        columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
        df_train = pd.DataFrame(frame, columns=columns)
        df_train['class'] = df_train['class'] - 1
        df_train['fname'] = ['./stanford_cars/cars_train/' + f for f in df_train['fname']]

        cars_test_annos = loadmat('./stanford_cars/cars_test_annos_withlabels.mat')
        frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]
        columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
        df_test = pd.DataFrame(frame, columns=columns)
        df_test['class'] = df_test['class'] - 1
        df_test['fname'] = ['./stanford_cars/cars_test/' + f for f in df_test['fname']]

        df_test = df_test.loc[df_test['class'] < 20]
        df_train = df_train.loc[df_train['class'] < 20]

        transform_train = transforms.Compose([transforms.Resize([image_size[0], image_size[1]]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.Resize([image_size[0], image_size[1]]),
                                             transforms.ToTensor()])

        if train_transform_cutpasted:
            train_set = Custom_Dataset(image_path=df_train["fname"].to_numpy(), targets=df_train["class"].to_numpy(),
                                       transform=train_transform_cutpasted)
        else:
            train_set = Custom_Dataset(image_path=df_train["fname"].to_numpy(), targets=df_train["class"].to_numpy(),
                                       transform=transform_train)
        test_set = Custom_Dataset(image_path=df_test["fname"].to_numpy(), targets=df_test["class"].to_numpy(),
                                  transform=transform_test)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

        print("len(test_set), len(train_set): ", len(test_set), len(train_set))
    elif dataset == 'cub-birds':
        n_classes = 2
        cub_root = 'cub'
        transform_train = transforms.Compose([transforms.Resize((image_size[0], image_size[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

        transform_test = transforms.Compose([transforms.Resize((image_size[0], image_size[1])),
                                             transforms.ToTensor()])
        normal_set = labels
        if train_transform_cutpasted:
            transform_train = train_transform_cutpasted
        train_set = CustomCub2011(root=cub_root, transform=transform_train, train=True)
        train_set = subsample_classes(train_set, include_classes=normal_set)
        train_set.target_transform = lambda x: 0

        anomaly_set = list(set(list(range(20))) - set(normal_set))

        test_set_in = CustomCub2011(root=cub_root, transform=transform_test, train=False)
        test_set_in = subsample_classes(test_set_in, include_classes=normal_set)
        test_set_in.target_transform = lambda x: 0

        test_set_out = CustomCub2011(root=cub_root, transform=transform_test, train=False)
        test_set_out = subsample_classes(test_set_out, include_classes=anomaly_set)
        test_set_out.target_transform = lambda x: 1

        test_set = ConcatDataset([test_set_in, test_set_out])

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        print("len(test_set), len(train_set): ", len(test_set), len(train_set))

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_pil':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_pil':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'svhn-10' or dataset == 'svhn-10-corruption' or dataset == "STL-10":
        return SVHN_SUPERCLASS
    elif dataset == 'cifar10-corruption':
        return CIFAR10_CORRUPTION_SUPERCLASS
    elif dataset == 'mnist-corruption':
        return MNIST_CORRUPTION_SUPERCLASS
    elif dataset == 'cifar10-versus-100':
        return CIFAR10_VER_CIFAR100_SUPERCLASS
    elif dataset == 'cifar100-versus-10':
        return CIFAR10_VER_CIFAR100_SUPERCLASS
    elif dataset == 'dtd':
        return DTD_SUPERCLASS
    elif dataset == "WBC":
        return WBC_SUPERCLASS
    elif dataset == 'breastmnist':
        return breastmnist_SUPERCLASS
    elif dataset == 'Tomor_Detection' or dataset == 'high-variational-brain-tumor':
        return TUMOR_BRAIN_SUPERCLASS
    elif dataset == 'MVTecAD':
        return MVTecAD_SUPERCLASS
    elif dataset == 'ArtBench':
        return ART_BENCH_SUPERCLASS
    elif dataset == 'cub-birds' or dataset == 'head-ct' or dataset == 'cifar100-versus-other-eval' or dataset == 'cifar10-versus-other-eval':
        return HEAD_CT_SUPERCLASS
    elif dataset == 'mvtec-high-var' or dataset == 'mvtec-high-var-corruption':
        return MVTEC_HV_SUPERCLASS
    elif dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'fashion-mnist':
        return FashionMNIST_SUPERCLASS
    elif dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'cifar100' or dataset == 'cifar100-corruption' or dataset == 'stanford-cars':
        return CIFAR100_SUPERCLASS
    elif dataset == 'ucsd':
        return UCSD_SUPERCLASS
    elif dataset == 'ISIC2018':
        return ISIC2018_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    elif dataset == 'dior':
        return DIOR_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes, count=-1):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    try:
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        # SVHN
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)

    dataset = Subset(dataset, indices)
    if count == -1:
        pass
    elif len(dataset) > count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset) - 1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num * len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset) - 1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset


def set_dataset_count(dataset, count=-1):
    if count == -1:
        pass
    elif len(dataset) > count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset) - 1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num * len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset) - 1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
