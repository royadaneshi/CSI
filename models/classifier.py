import torch.nn as nn

from models.resnet import ResNet18, ResNet34, ResNet50, Pretrain_ResNet18_Model, Pretrain_ResNet152_Model, Pretrain_ResNet152_Corruption_Model, Pretrain_ResNet18_Corruption_Model, Pretrain_Wide_ResNet_Model, Pretrain_ConvNext_Model
from models.resnet_imagenet import resnet18, resnet50
from models.wide_resnet import wide_resnet34_5
import models.transform_layers as TL
from models.vit import VIT_Pretrain
from models.vit_FITYMI import VIT_Pretrain_FITYMI

def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    if P.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform


def get_shift_module(P, eval=False):

    if P.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4
    elif P.shift_trans_type == 'cutperm':
        shift_transform = TL.CutPerm()
        K_shift = 4
    else:
        shift_transform = nn.Identity()
        K_shift = 1

    if not eval and not ('sup' in P.mode):
        assert P.batch_size == int(128/K_shift)

    return shift_transform, K_shift


def get_shift_classifer(model, K_shift):

    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)

    return model


def get_classifier(mode, n_classes=10, activation='relu', std=1.0, mean=0.0, noise_scale=0.1, noist_probability=0.5):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes, activation=activation)
    elif mode == 'resnet18-corruption':
        classifier = Pretrain_ResNet18_Corruption_Model(num_classes=n_classes, std=std, mean=mean, noise_scale=noise_scale, probability=noist_probability)
    elif mode == "vit_fitymi":
        classifier = VIT_Pretrain_FITYMI(num_classes=n_classes)
    elif mode == "pretrain-wide-resnet":
        classifier = Pretrain_Wide_ResNet_Model(num_classes=n_classes)
    elif mode == "vit":
        classifier = VIT_Pretrain(num_classes=n_classes)
    elif mode == 'pretrain-resnet152-corruption':
        classifier = Pretrain_ResNet152_Corruption_Model(num_classes=n_classes)
    elif mode =='pretrain-resnet152':
        classifier = Pretrain_ResNet152_Model(num_classes=n_classes)
    elif mode =='pretrain-resnet18':
        classifier = Pretrain_ResNet18_Model(num_classes=n_classes)
    elif mode == 'resnet34':
        classifier = ResNet34(num_classes=n_classes)
    elif mode == 'resnet50':
        classifier = ResNet50(num_classes=n_classes)
    elif mode == 'resnet18_imagenet':
        classifier = resnet18(num_classes=n_classes)
    elif mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
    elif mode == 'wide_resnet34_5':
        classifier = wide_resnet34_5(num_classes=n_classes)
    elif mode == 'conv_next':
        classifier = Pretrain_ConvNext_Model(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

