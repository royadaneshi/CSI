import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils_.utils import set_random_seed
from datasets.cutpast_transformation import *
from PIL import Image
from glob import glob
import pickle
import random
import rasterio
import re
from torchvision.datasets.folder import default_loader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision
import subprocess
from tqdm import tqdm
import requests
import shutil
from PIL import Image
import shutil
import random
import zipfile
import time
import gdown

CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut',
               'hazelnut', 'screw', 'carpet', 'leather', 'cable']
DATA_PATH = './data/'


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, 'train', "*", "images", "*.JPEG"))
        if count == -1:
            final_length = len(image_files)
        else:
            random.shuffle(image_files)
            final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


class FakeMVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category FakeMVTecDataset:", category)
        self.image_files = glob(os.path.join(root, category, "*.jpeg"))
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class MVTecDataset_Cutpasted(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset_Cutpasted:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class DataOnlyDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx][0]
        return sample


class HEAD_CT_DATASET(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class FakeCIFAR10(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class FakeMNIST(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=6000):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = list(np.load("./Fake_Mnist.npy")[6000 * i:6000 * (i + 1)])
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        # self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image = Image.fromarray((self.image_files[index].transpose(1, 2, 0) * 255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


class FakeWBC(Dataset):
    def __init__(self, root='./', category=0, transform=None, count=6000):
        self.transform = transform
        self.image_files = list(np.load(root + "Fake_CELL_type1.npy"))
        if count < len(self.image_files):
            self.image_files = self.image_files[:count]
        else:
            t = len(self.image_files)
            for i in range(count - t):
                self.image_files.append(random.choice(self.image_files[:t]))

    def __getitem__(self, index):
        image = Image.fromarray((self.image_files[index].transpose(1, 2, 0) * 255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


class FakeFashionDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class Fake_SVHN_Dataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = glob(os.path.join(root, str(category[i]), "*.jpeg"))
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


class MVTecDataset_High_VAR(Dataset):
    def __init__(
            self,
            dataset_path="./mvtec_anomaly_detection",
            class_name="bottle",
            is_train=True,
            resize=256,
            cropsize=224,
            transform=None,
    ):
        assert class_name in CLASS_NAMES, "class_name: {}, should be in {}".format(
            class_name, CLASS_NAMES
        )
        print("class_name MVTecDataset_High_VAR:", class_name)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        self.dataset_path = os.path.join(dataset_path, "mvtec_anomaly_detection")
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if transform:
            self.transform_x = transform
        else:
            self.transform_x = transforms.Compose(
                [
                    transforms.Resize(resize, Image.ANTIALIAS),
                    transforms.CenterCrop(cropsize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST), transforms.CenterCrop(cropsize), transforms.ToTensor()]
        )

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)
        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"
        return list(x), list(y), list(mask)


class FakeCIFAR100(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=2500):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = list(np.load("./cifar100_training_gen_data.npy")[2500 * i:2500 * (i + 1)])
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        # self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image = Image.fromarray(self.image_files[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class FakeDTD(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, count=2500):
        self.transform = transform
        self.image_files = []
        for i in range(len(category)):
            img_files = list(np.load(f"./DTD_{category[i]}_X.npy"))
            if count[i] < len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i] - t):
                    img_files.append(random.choice(img_files[:t]))
            self.image_files += img_files
        '''
        for i in range(len(category)):
            img_files = list(np.load("./cifar100_training_gen_data.npy")[2500*i:2500*(i+1)])
            if count[i]<len(img_files):
                img_files = img_files[:count[i]]
            else:
                t = len(img_files)
                for i in range(count[i]-t):
                    img_files.append(random.choice(img_files[:t]))            
            self.image_files += img_files
        '''

    def __getitem__(self, index):
        image = Image.fromarray((self.image_files[index].transpose(1, 2, 0) * 255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)


class UCSDDataset(Dataset):
    def __init__(self, root, dataset='ped1', is_normal=True, transform=None, target_transform=None, download=False):
        self.root = os.path.join(root, 'UCSD_Anomaly_Dataset.v1p2')
        self.is_normal = is_normal
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        # download not supported

        if self.dataset == 'ped1':
            base_dir = 'UCSDped1'
        if self.dataset == 'ped2':
            base_dir = 'UCSDped2'

        if not self.is_normal:
            sub_dir = 'Test'
        else:
            sub_dir = 'Train'

        video_dir = glob(os.path.join(self.root, base_dir, sub_dir, sub_dir + '*'))
        self.video_dir = sorted([x for x in video_dir if re.fullmatch('.*\d\d\d', x)])
        self.videos_len = []
        self.images_dir = []
        for video in self.video_dir:
            images = list(sorted(glob(os.path.join(video, "*.tif"))))
            self.images_dir += images
            self.videos_len.append(len(images))
        self.num_samples = len(self.images_dir)
        self.labels = self._gather_labels()

    def __getitem__(self, index):
        with rasterio.open(self.images_dir[index]) as image:
            image_array = image.read()
            # torch.Size([238, 1, 158])
            image = transforms.ToPILImage(mode='RGB')(
                transforms.ToTensor()(image_array).permute(1, 2, 0).repeat(3, 1, 1)
            )
        if self.transform:
            image = self.transform(image)
        label = self.get_label(index)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _gather_labels(self):
        if self.is_normal:
            return None
        if self.dataset == 'ped1':
            base_dir = 'UCSDped1'
        if self.dataset == 'ped2':
            base_dir = 'UCSDped2'

        with open(os.path.join(self.root, base_dir, 'Test', f'{base_dir}.m'), 'r') as file:
            lines = file.readlines()

        annotations = []

        video_index = 0
        # Iterate over the lines
        for line in lines:
            # Use regular expressions to extract the frame ranges
            matches = re.findall(r'(\d+:\d+)', line)
            if len(matches) == 0:
                continue

            frame_mask = np.zeros(self.videos_len[video_index], dtype=bool)
            for match in matches:
                start, end = map(int, match.split(':'))
                frame_mask[start - 1:end] = True

            annotations.append(frame_mask)
            video_index += 1
        annotations = np.concatenate(annotations)
        return annotations

    def get_label(self, index):
        if self.is_normal:
            label = 0
        else:
            label = 1 if self.labels[index] else 0

        return label

    def __len__(self):
        return len(self.images_dir)


####################################33

class chest(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True, count=-1):
        self.transform = transform
        self.train = train
        self.image_files = []

        if train:
            self.image_files = glob(os.path.join('/kaggle/input/chest-datasett256/chest_dataset/train', '*.png'))
        else:
            self.image_files = glob(os.path.join('/kaggle/input/chest-datasett256/chest_dataset/test', '*.png'))

        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                if t > 0:
                    for i in range(count - t):
                        self.image_files.append(random.choice(self.image_files[:t]))

        self.image_files.sort(key=lambda y: y.lower())
    def __getitem__(self, index):
        # print(len(self.image_files), "---", index)
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((256, 256))

        if self.transform:
            image = self.transform(image)
        target = 0
        if "normal" in os.path.dirname(image_file):
            target = 0
        elif "abnormal" in os.path.dirname(image_file):
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


##############################################################


class TumorDetection(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True, count=None):
        self._download_and_extract()
        self.transform = transform
        if train:
            self.image_files = glob(os.path.join('./MRI', "Training", "notumor", "*.jpg"))
        else:
            image_files = glob(os.path.join('./MRI', "Testing", "*", "*.jpg"))
            normal_image_files = glob(os.path.join('./MRI', "./Testing", "notumor", "*.jpg"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files

        if count is not None:
            if count > len(self.image_files):
                self.image_files = self._oversample(count)
            else:
                self.image_files = self._undersample(count)

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def _download_and_extract(self):
        google_id = '1AOPOfQ05aSrr2RkILipGmEkgLDrZCKz_'
        file_path = os.path.join('./MRI', 'Training')

        if os.path.exists(file_path):
            return

        if not os.path.exists('./MRI'):
            os.makedirs('./MRI')

        if not os.path.exists(file_path):
            subprocess.run(['gdown', google_id, '-O', './MRI/archive(3).zip'])

        with zipfile.ZipFile("./MRI/archive(3).zip", 'r') as zip_ref:
            zip_ref.extractall("./MRI/")

        os.rename("./MRI/Training/glioma", "./MRI/Training/glioma_tr")
        os.rename("./MRI/Training/meningioma", "./MRI/Training/meningioma_tr")
        os.rename("./MRI/Training/pituitary", "./MRI/Training/pituitary_tr")

        shutil.move("./MRI/Training/glioma_tr", "./MRI/Testing")
        shutil.move("./MRI/Training/meningioma_tr", "./MRI/Testing")
        shutil.move("./MRI/Training/pituitary_tr", "./MRI/Testing")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((256, 256))

        if self.transform:
            image = self.transform(image)

        if "notumor" in os.path.dirname(image_file):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)

    def _oversample(self, count):
        num_extra_samples = count - len(self.image_files)
        extra_image_files = [random.choice(self.image_files) for _ in range(num_extra_samples)]

        return self.image_files + extra_image_files

    def _undersample(self, count):
        indices = random.sample(range(len(self.image_files)), count)
        new_image_files = [self.image_files[idx] for idx in indices]

        return new_image_files


class BrainMRI(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]

            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class AdaptiveExposure(Dataset):
    def __init__(self, root, transform, count=None):
        super(AdaptiveExposure, self).__init__()
        self.root = root
        self.image_files = glob(os.path.join(root, '**', "*.png"), recursive=True)
        self.transform = transform
        if count is not None:
            if count > len(self.image_files):
                self.image_files = self._oversample(count)
            else:
                self.image_files = self._undersample(count)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((256, 256))

        if self.transform:
            image = self.transform(image)

        return image, 1

    def _oversample(self, count):
        num_extra_samples = count - len(self.image_files)
        extra_image_files = [random.choice(self.image_files) for _ in range(num_extra_samples)]

        return self.image_files + extra_image_files

    def _undersample(self, count):
        indices = random.sample(range(len(self.image_files)), count)
        new_image_files = [self.image_files[idx] for idx in indices]

        return new_image_files


class HEAD_CT_FAKE(Dataset):
    def __init__(self, transform=None, count=6000):
        self.transform = transform
        self.image_files = list(np.load("./Head-CT-50.npy"))
        if count < len(self.image_files):
            self.image_files = self.image_files[:count]
        else:
            t = len(self.image_files)
            for i in range(count - t):
                self.image_files.append(random.choice(self.image_files[:t]))

    def __getitem__(self, index):
        image = Image.fromarray((self.image_files[index].transpose(1, 2, 0) * 255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


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


class CIFAR_CORRUCPION(Dataset):
    def __init__(self, transform=None, normal_idx=[0], cifar_corruption_label='CIFAR-10-C/labels.npy',
                 cifar_corruption_data='./CIFAR-10-C/defocus_blur.npy'):
        self.labels_10 = np.load(cifar_corruption_label)
        self.labels_10 = self.labels_10[:10000]
        if cifar_corruption_label == 'CIFAR-100-C/labels.npy':
            self.labels_10 = sparse2coarse(self.labels_10)
        self.data = np.load(cifar_corruption_data)
        self.data = self.data[:10000]
        self.transform = transform
        '''
        def indice_by_label(data_labels, normal_labels):
            items_index = []
            for label in normal_labels:
                items_index = items_index + list(np.where(self.labels_10 == 1)[0])
            return items_index
        
        normal_indice = indice_by_label(data_labels=self.labels_10, normal_labels=normal_idx)
        anomaly_indice = list(set(list(range(len(self.labels_10)))) - set(normal_indice))

        self.labels_10[normal_indice] = 0
        self.labels_10[anomaly_indice] = 1
        '''

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels_10[index]
        if self.transform:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class MNIST_CORRUPTION(Dataset):
    def __init__(self, root_dir, corruption_type, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.corruption_type = corruption_type
        self.train = train

        indicator = 'train' if train else 'test'
        folder = os.path.join(self.root_dir, self.corruption_type, f'saved_{indicator}_images')
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        if train:
            data = np.load(os.path.join(root_dir, corruption_type, 'train_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'train_labels.npy'))
        else:
            data = np.load(os.path.join(root_dir, corruption_type, 'test_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'test_labels.npy'))

        self.labels = labels
        self.image_paths = []

        for idx, img in enumerate(data):
            path = os.path.join(folder, f"{idx}.png")
            self.image_paths.append(path)

            if not os.path.exists(path):
                img_pil = torchvision.transforms.ToPILImage()(img)
                img_pil.save(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class MyDataset_Binary(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, labels, transform=None, cutpast_transformation=None):
        'Initialization'
        self.labels = labels
        self.x = x
        self.transform = transform
        self.cutpast_transformation = cutpast_transformation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.cutpast_transformation:
            # print(self.x[index])
            # print(self.labels[index])
            # x = Image.fromarray((np.array(self.x[index]).transpose(1, 2, 0) * 255).astype(np.uint8))
            x = self.cutpast_transformation(x)
            y = self.labels[index]
        elif self.transform is None:
            x = self.x[index]
            y = self.labels[index]
        else:
            x = self.transform(self.x[index])
            y = self.labels[index]
        return x, y


class DIOR(Dataset):
    links = {
        "train": [
            "https://drive.google.com/file/d/1--NeRTtWINde8GURrstElpL0OtLhR80J/view?usp=sharing",
            "train.zip",
            "DIOR"
        ],
        "test": [
            "https://drive.google.com/file/d/1-3J5vJvzn24Aj2thEQm_qyrMv1PjDas4/view?usp=sharing",
            "test.zip",
            "DIOR"
        ]
    }

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, count=-1, verbose=False,
                 **kwargs):
        super(DIOR, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.count = count
        if download:
            self._download_and_extract()
        self.data, self.targets, self.classes = self._load_data()
        self._balance_data()

    def _download_and_extract(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if self.train:
            link = self.links["train"]
        else:
            link = self.links["test"]

        file_path = os.path.join(self.root, link[1])

        if not os.path.exists(file_path):
            gdown.download(link[0], file_path, quiet=not self.verbose, fuzzy=True)

        data_path = os.path.join(self.root, link[2])
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

    def _load_data(self):
        if self.train:
            data_path = os.path.join(self.root, self.links["train"][2])
        else:
            data_path = os.path.join(self.root, self.links["test"][2])
        with open(os.path.join(data_path, 'class_names.pkl'), 'rb') as f:
            classes = pickle.load(f)
        targets_name = f"{('train' if self.train else 'test')}_targets.pkl"
        with open(os.path.join(data_path, targets_name), 'rb') as f:
            targets = pickle.load(f)
        img_dir = os.path.join(data_path, 'train' if self.train else 'test')
        data = [os.path.join(img_dir, f'{idx:0>5}.jpg') for idx in range(len(targets))]
        return data, targets, classes

    def _balance_data(self):
        if self.count == -1:
            return
        # TODO: Implement data balancing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]
        target = self.targets[index]
        img = Image.open(image_name)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class DIOR_OOD(DIOR):

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, count=-1, verbose=False,
                 normal_classes=0, **kwargs):
        super().__init__(root, train, download, transform, target_transform, count, verbose, **kwargs)
        if not isinstance(normal_classes, list):
            normal_classes = [normal_classes]
        self.normal_classes = normal_classes
        if self.verbose:
            print(f'Normal classes: {[self.classes[i] for i in normal_classes]}')

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if target in self.normal_classes:
            target = 0
        else:
            target = 1
        return img, target


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Resize([224, 224])
    dior_ood = DIOR_OOD(root='.', train=False, download=False, normal_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17],
                        verbose=True, transform=transform)

    # Visualize some data points
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    rand_idx = np.random.choice(range(len(dior_ood)), size=16, replace=False)
    samples = [dior_ood[idx] for idx in rand_idx]
    for i in range(4):
        for j in range(4):
            image, target = samples[4 * i + j]
            axs[i, j].imshow(image)
            axs[i, j].title.set_text(target)
    plt.show()


class WBC_DATASET(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, train=True, transform=None, normal_set=[1, 2], path="./CELL_MIR/trainig/"):
        'Initialization'
        np.random.seed(47)
        df = pd.read_csv("./segmentation_WBC/Class Labels of Dataset 1.csv")
        self.labels = df['class label'].to_numpy()
        data_id = df['image ID'].to_numpy()
        self.data_path = np.array([(path + str("%03d" % p) + '.png') for p in data_id])

        self.labels = self.labels - 1
        self.data_path = self.data_path

        training_normal = []
        test_normal = []
        for normal in normal_set:
            normal_idx = []
            for i, l in enumerate(self.labels):
                if l in [normal]:
                    normal_idx.append(i)
            np.random.shuffle(normal_idx)
            training_normal = training_normal + normal_idx[:int(len(normal_idx) * 0.8)]
            test_normal = test_normal + normal_idx[int(len(normal_idx) * 0.8):]
        print("number of training_normal:", len(training_normal))
        print("number of test_normal:", len(test_normal))

        # print("1 len(self.labels), len(self.data_path)", len(self.labels), len(self.data_path))
        if train:
            self.labels[training_normal] = 0
            self.labels = self.labels[training_normal]
            self.data_path = self.data_path[training_normal]
        else:
            test_anomaly = np.arange(300)
            test_anomaly = np.setdiff1d(test_anomaly, training_normal + test_normal)
            print("number of test_anomaly:", len(test_anomaly))

            self.labels[test_anomaly] = 1
            self.labels[test_normal] = 0
            test_idx = np.concatenate((test_anomaly, test_normal)).astype(int)
            self.labels = self.labels[test_idx]
            self.data_path = self.data_path[test_idx]

        # print("len(self.labels), len(self.data_path)", len(self.labels), len(self.data_path))
        # print(list(zip(self.data_path, self.labels)))
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        image_file = self.data_path[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


from torchvision.datasets import CIFAR10


class ArtBench10(CIFAR10):
    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }


class ISIC2018(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class ImageNet30_Dataset(Dataset):
    def __init__(self, image_path, labels, transform=None):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class Custom_Dataset(Dataset):
    def __init__(self, image_path, targets, transform=None):
        self.transform = transform
        self.image_files = image_path
        self.targets = targets

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.image_files)


class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):
    include_classes_cub = np.array(include_classes) + 1  # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset
