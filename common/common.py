from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['chest','STL-10', 'cub-birds', 'stanford-cars', 'cifar100-versus-other-eval', 'cifar10-versus-other-eval', 'ISIC2018', 'ArtBench', 'high-variational-brain-tumor', 'svhn-10-corruption', 'mvtec-high-var-corruption', 'cifar100-corruption', 'cifar100-versus-10', 'cifar10-versus-100', 'WBC', 'dtd', 'cifar10-corruption', 'mnist-corruption', 'Tomor_Detection', 'ucsd', 'mvtec-high-var', 'breastmnist', 'head-ct', 'fashion-mnist', 'mnist', 'cifar10', 'cifar100', 'imagenet', 'svhn-10', 'MVTecAD', 'dior'],
                        default="cifar10", type=str)
    parser.add_argument('--outlier_dataset', help='',
                            default="mnist",choices=['imagenet30', 'mnist', 'svhn', 'fashion-mnist'] ,type=str)
    parser.add_argument('--normal_labels', help='normal_labels for high variation',
                        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13", type=str)
    parser.add_argument('--cifar_corruption_data', help='',
                        default="./CIFAR-10-C/defocus_blur.npy", type=str)
    parser.add_argument('--mnist_corruption_folder', help='',
                        default="./mnist_c/", type=str)
    parser.add_argument('--mnist_corruption_type', help='MNIST corruption type',
                        choices=[
                            "brightness",
                            "canny_edges",
                            "dotted_line",
                            "fog",
                            "glass_blur",
                            "identity",
                            "impulse_noise",
                            "motion_blur",
                            "rotate",
                            "scale",
                            "shear",
                            "shot_noise",
                            "spatter",
                            "stripe",
                            "translate",
                            "zigzag"
                        ],
                        default="brightness", type=str)
                
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--unfreeze_pretrain_model_epoch', help='unfreeze_pretrain_model',
                        default=50, type=int)
    parser.add_argument('--image_size', help='None: multi-class, Not None: one-class',
                        default=32, type=int)
    parser.add_argument('--save_step', help='None: multi-class, Not None: one-class',
                        default=20, type=int) 

    parser.add_argument('--noise_mean', help='',
                        default=0.0, type=float)
    parser.add_argument('--noise_std', help='',
                        default=1.0, type=float)
    parser.add_argument('--noise_scale', help='',
                        default=0.1, type=float)
    parser.add_argument('--noist_probability', help='',
                        default=0.5, type=float)                        
    parser.add_argument('--fake_data_percent', help='',
                        default=0.0, type=float)
    parser.add_argument('--cutpast_data_percent', help='',
                        default=0.0, type=float)
    parser.add_argument('--main_count', help='count of normal data',
                        default=-1, type=int)
    parser.add_argument('--high_var', help='not used!',
                        default=0, choices=[0, 1], type=int)
    parser.add_argument('--activation_function', help='activation_function for resnet from scratch model.(note this argument is used just in resent18 from scratch)',
                        choices=['relu', 'gelu'], default="relu", type=str)
    parser.add_argument('--model', help='Model',
                        choices=['clip_vit', 'clip_r50', 'R50ViT', 'dino', 'conv_next', 'pretrain-wide-resnet', 'resnet18-corruption', 'pretrain-resnet152-corruption', 'pretrain-resnet152', 'vit_fitymi', 'vit', 'resnet18', 'resnet18_imagenet', 'pretrain-resnet18', 'wide_resnet34_5'], default="resnet18", type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='simclr', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')
    parser.add_argument("--cl_no_hflip", help='activate to not used hflip in contrastive augmentaion.',
                        action='store_true')
    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--print_3_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')
    parser.add_argument('--timer', default=None, type=int)

    parser.add_argument('--freezing_layer', help='Freezing Layer',
                        default=133, type=int)
    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
