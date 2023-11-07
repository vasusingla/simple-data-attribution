# -*- coding: utf-8 -*-
"""
Benchmark Results

Updated: 18.02.2022 (6618fa3c36b0c9f3a9d7a21bcdb00bf4fd258ee8))

------------------------------------------------------------------------------------------
| Model         | Batch Size | Epochs |  KNN Test Accuracy |       Time | Peak GPU Usage |
------------------------------------------------------------------------------------------
| BarlowTwins   |        128 |    200 |              0.835 |  193.4 Min |      2.2 GByte |
| BYOL          |        128 |    200 |              0.872 |  217.0 Min |      2.3 GByte |
| DCL (*)       |        128 |    200 |              0.842 |  126.9 Min |      1.7 GByte |
| DCLW (*)      |        128 |    200 |              0.833 |  127.5 Min |      1.8 GByte |
| DINO          |        128 |    200 |              0.868 |  220.7 Min |      2.3 GByte |
| Moco          |        128 |    200 |              0.838 |  229.5 Min |      2.3 GByte |
| NNCLR         |        128 |    200 |              0.838 |  198.7 Min |      2.2 GByte |
| SimCLR        |        128 |    200 |              0.822 |  182.7 Min |      2.2 GByte |
| SimSiam       |        128 |    200 |              0.779 |  182.6 Min |      2.3 GByte |
| SwaV          |        128 |    200 |              0.806 |  182.4 Min |      2.2 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    200 |              0.827 |  160.7 Min |      7.5 GByte |
| BYOL          |        512 |    200 |              0.872 |  188.5 Min |      7.7 GByte |
| DCL (*)       |        512 |    200 |              0.834 |  113.6 Min |      6.1 GByte |
| DCLW (*)      |        512 |    200 |              0.830 |  113.8 Min |      6.2 GByte | 
| DINO          |        512 |    200 |              0.862 |  191.1 Min |      7.5 GByte |
| Moco (**)     |        512 |    200 |              0.850 |  196.8 Min |      7.8 GByte |
| NNCLR (**)    |        512 |    200 |              0.836 |  164.7 Min |      7.6 GByte |
| SimCLR        |        512 |    200 |              0.828 |  158.2 Min |      7.5 GByte |
| SimSiam       |        512 |    200 |              0.814 |  159.0 Min |      7.6 GByte |
| SwaV          |        512 |    200 |              0.833 |  158.4 Min |      7.5 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    800 |              0.857 |  641.5 Min |      7.5 GByte |
| BYOL          |        512 |    800 |              0.911 |  754.2 Min |      7.8 GByte |
| DCL (*)       |        512 |    800 |              0.873 |  459.6 Min |      6.1 GByte |
| DCLW (*)      |        512 |    800 |              0.873 |  455.8 Min |      6.1 GByte | 
| DINO          |        512 |    800 |              0.884 |  765.5 Min |      7.6 GByte |
| Moco (**)     |        512 |    800 |              0.900 |  787.7 Min |      7.8 GByte |
| NNCLR (**)    |        512 |    800 |              0.896 |  659.2 Min |      7.6 GByte |
| SimCLR        |        512 |    800 |              0.875 |  632.5 Min |      7.5 GByte |
| SimSiam       |        512 |    800 |              0.906 |  636.5 Min |      7.6 GByte |
| SwaV          |        512 |    800 |              0.881 |  634.9 Min |      7.5 GByte |
------------------------------------------------------------------------------------------

(*): Smaller runtime and memory requirements due to different hardware settings
and pytorch version. Runtime and memory requirements are comparable to SimCLR
with the default settings.
(**): Increased size of memory bank from 4096 to 8192 to avoid too quickly 
changing memory bank due to larger batch size.

The benchmarks were created on a single NVIDIA RTX A6000.

Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)

"""
import copy
import os

import time
import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
import argparse

parser = argparse.ArgumentParser(description='Benchmark SSL models')
parser.add_argument('--model', type=str, default='SimCLR', help='SSL model to benchmark')
parser.add_argument('--backbone', type=str, default='resnet-18', help='Model backbone')
args = parser.parse_args()

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
num_workers = 8
knn_k = 200
knn_t = 0.1
classes = 10

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True). 
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating 
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all 
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False 

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_size = 512
lr_factor = batch_size / 128 # scales the learning rate linearly with batch size

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if distributed:
    distributed_backend = 'ddp'
    # reduce batch size for distributed training
    batch_size = batch_size // gpus
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle 
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = os.path.join(os.getenv('NEXUS_DIR'), 'datasets/cifar10/cifar10/train/')
path_to_test = os.path.join(os.getenv('NEXUS_DIR'), 'datasets/cifar10/cifar10/test/')

# Use SimCLR augmentations, additionally, disable blur for cifar10
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
swav_collate_fn = lightly.data.SwaVCollateFunction(
    crop_sizes=[32],
    crop_counts=[2], # 2 crops @ 32x32px
    crop_min_scales=[0.14],
    gaussian_blur=0,
)

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_collate_fn = lightly.data.DINOCollateFunction(
    global_crop_size=32,
    n_local_views=0,
    gaussian_blur=(0, 0, 0),
)

# Two crops for SMoG
smog_collate_function = lightly.data.collate.SMoGCollateFunction(
    crop_sizes=[32, 32],
    crop_counts=[1, 1],
    gaussian_blur_probs=[0., 0.],
    crop_min_scales=[0.2, 0.2],
    crop_max_scales=[1.0, 1.0],
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset(
    input_dir=path_to_train
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train,
    transform=test_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)

def get_data_loaders(batch_size: int, model):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = collate_fn
    if model == SwaVModel:
        col_fn = swav_collate_fn
    elif model == DINOModel:
        col_fn = dino_collate_fn
    elif model == SMoGModel:
        col_fn = smog_collate_function
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test



models = [
    eval(args.model)
    # BarlowTwinsModel,
    # BYOLModel,
    # DCL,
    # DCLW,
    # DINOModel,
    # MocoModel,
    # NNCLRModel,
    # SimCLRModel,
    # SimSiamModel,
    # SwaVModel,
    # SMoGModel
]
bench_results = dict()

experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace('Model', '')
    for seed in range(n_runs):
        pl.seed_everything(seed)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, 
            model=BenchmarkModel,
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes,
                                        lr_factor, max_epochs, backbone = args.backbone)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, 'cifar10'),
            name='',
            sub_dir=sub_dir,
            version=experiment_version,
        )
        if experiment_version is None:
            # Save results of all models under same version directory
            experiment_version = logger.version
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, 'checkpoints')
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs, 
            gpus=gpus,
            default_root_dir=logs_root_dir,
            strategy=distributed_backend,
            sync_batchnorm=sync_batchnorm,
            logger=logger,
            callbacks=[checkpoint_callback]
        )
        start = time.time()
        trainer.fit(
            benchmark_model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test
        )
        end = time.time()
        run = {
            'model': model_name,
            'batch_size': batch_size,
            'epochs': max_epochs,
            'max_accuracy': benchmark_model.max_accuracy,
            'runtime': end - start,
            'gpu_memory_usage': torch.cuda.max_memory_allocated(),
            'seed': seed,
        }
        runs.append(run)
        print(run)

        # delete model and trainer + free up cuda memory
        del benchmark_model
        del trainer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    bench_results[model_name] = runs

# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print('-' * len(header))
print(header)
print('-' * len(header))
for model, results in bench_results.items():
    runtime = np.array([result['runtime'] for result in results])
    runtime = runtime.mean() / 60 # convert to min
    accuracy = np.array([result['max_accuracy'] for result in results])
    gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True
    )
print('-' * len(header))
