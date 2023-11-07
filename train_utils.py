
import os
import torch as ch
import torchvision
import numpy as np
from tqdm import tqdm
import time
import argparse
import ipdb

from typing import List, Callable, Tuple, Optional
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from ffcv.pipeline.state import State
from ffcv.pipeline.operation import AllocationQuery, Operation
from ffcv.pipeline.compiler import Compiler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from models.resnet import resnet9
from models.swin import swin_t
from models.mobilenetv2 import MobileNetV2
from constants import CIFAR_MEAN, CIFAR_STD

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=512)
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--epochs', type=int, default=24)
#     parser.add_argument('--train-samples-remove')

class CorruptFixedLabels(Operation):
    def __init__(self, flip_class, corrupt_idxs=None):
        super().__init__()
        self.flip_class = flip_class
        self.corrupt_idxs = corrupt_idxs

    def generate_code(self) -> Callable:
        # dst will be None since we don't ask for an allocation
        parallel_range = Compiler.get_iterator()
        corrupt_idxs = self.corrupt_idxs
        def corrupt_fixed(labs, _, inds):
            for iter_idx in parallel_range(labs.shape[0]):
                dset_idx = inds[iter_idx]
                if dset_idx in corrupt_idxs:
                    labs[iter_idx] = self.flip_class
                # if np.random.rand() < 0.2:
                    # They will also be corrupted to a deterministic label:
                    # labs[i] = np.random.randint(low=0, high=10)
            return labs

        corrupt_fixed.is_parallel = True
        corrupt_fixed.with_indices = True
        return corrupt_fixed

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # No updates to state or extra memory necessary!
        return previous_state, None

def create_model(arch='resnet9'):
    if arch == 'resnet9':
        model = resnet9(num_classes=10)
    elif arch == 'swin_t':
        model = swin_t(window_size=4,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1),)
    elif arch == 'mobilenetv2':
        model = MobileNetV2(num_classes=10)
    else:
        print(f"Model {arch} not supported.")
        raise NotImplementedError
    model = model.to(memory_format=ch.channels_last).cuda()
    return model

def create_loaders(train_path=None, train_examples_remove=None, 
                  eval_idx=None, batch_size=512, num_workers=4,
                  flip_class=None):
    loaders = {}
    for name in ['train', 'test']:
        if name == 'train' and flip_class is not None:
            label_pipeline: List[Operation] = [IntDecoder(), CorruptFixedLabels(), ToTensor(), ToDevice('cuda:0')]
        else:
            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0')]
        if name == 'train':
            label_pipeline.extend([Squeeze()])
        else:
            if eval_idx is not None:
                label_pipeline.extend([Squeeze(0)])
            else:
                label_pipeline.extend([Squeeze()])
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        if name == 'train':
            selected_idx = set(np.arange(50000))
            if train_examples_remove is not None:
                train_examples_remove = set(train_examples_remove)
                selected_idx -= train_examples_remove
            selected_idx = list(selected_idx)
        else:
            if eval_idx is not None:
                selected_idx = [eval_idx]
            else:
                selected_idx = list(np.arange(10000))
        order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
        if name == 'train' and train_path is None:
            path = os.path.join(os.getenv('DATA_DIR'), 'cifar_train.beton')
        elif name=='test':
            path = os.path.join(os.getenv('DATA_DIR'), 'cifar_test.beton')
        else:
            path = train_path
        loaders[name] = Loader(path,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                order=order,
                                drop_last=(name == 'train'),
                                indices=selected_idx,
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})
    return loaders

def train_slow(model, loaders, batch_size=512, epochs=24, lr=0.1):
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = 50000 // batch_size
    scheduler = lr_scheduler.MultiStepLR(opt, 
                        milestones=[0.5 * iters_per_epoch, 
                                    0.75 * iters_per_epoch], 
                        gamma=0.1)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(epochs):
        for ims, labs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()


def train(model, loaders, batch_size, epochs=24):
    opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = 50000 // batch_size
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(epochs):
        for ims, labs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

def test(model, loaders):
    model.eval()
    with ch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in loaders['test']:
            with autocast():
                out = (model(ims))
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
    return total_correct, total_num

def create_train_dataset(train_idxs, flip_class, eval_idx):
    # Create new datasets with flipped labels.
    train_data = torchvision.datasets.CIFAR10(os.getenv("DATA_DIR"), train=True, download=False)

    # Flip training data labels, based on the given indexes.
    for idx in train_idxs:
        train_data.targets[idx] = flip_class

    train_path = f'/scratch0/cifar_train_{eval_idx}.beton'
    writer = DatasetWriter(train_path, {
                'image': RGBImageField(),
                'label': IntField()
            })
    writer.from_indexed_dataset(train_data)
    return train_path

def counterfactual_test(train_idxs=None, flip_class=None,
                        eval_idx=None, num_tests = 10, batch_size=512,
                        epochs=24, num_workers=4,
                        arch='resnet9'):
    # Remove train examples, and test if prediction flips on certain test samples.
    # Run the test multiple times to get a better estimate.

    # For training, load indexes to remove from the train set.
    # For evaluation, only use the given eval idx.
    # ipdb.set_trace()
    total_correct = 0
    if flip_class is None:
        loaders = create_loaders(train_examples_remove=train_idxs, 
                                eval_idx=eval_idx, batch_size=batch_size,
                                num_workers=num_workers)
    else:
        train_path = create_train_dataset(train_idxs, flip_class, eval_idx) 
        loaders = create_loaders(train_path=train_path, 
                                eval_idx=eval_idx, batch_size=batch_size,
                                num_workers=num_workers)
    for test_num in range(num_tests):
        model = create_model(arch=arch)
        train(model, loaders, batch_size, epochs=epochs)
        correct, _ = test(model, loaders)
        total_correct += correct
    return total_correct/num_tests

def binary_search(train_idxs, flip_class=None,
                 eval_idx=None, search_budget = 8, num_tests = 5,
                 arch='resnet9'):
    # Fixed budget binary search for the minimum number of examples to remove.
    # Note that this is not guaranteed to find the minimum.
    # It is possible that the minimum is not in the range [0, len(train_examples_remove)].
    # If we can't flip with all selected training examples, return -1.
    # If we can flip it, minimize the number of examples to remove using binary search.
    # We could use a linear search, but that would be too slow.
    # The binary search idea is supported by monotonicity of train-test influence.
    # This monotonicity is also supported by datamodels being linear.

    num_search = search_budget
    # Get the number of examples to remove for each binary search step.
    low = 0
    high = len(train_idxs) 
    mid = high
 
    avg_correct = counterfactual_test(train_idxs[:mid], flip_class, eval_idx, num_tests=num_tests,
                                      arch=arch)
    if avg_correct > 0.5:
        print(f"Sample Idx: {eval_idx}, Min samples: -1")
        return -1 # if can't flip with all selected training examples, return -1
    
    min_samples = mid
    while num_search > 0:
        num_search -= 1
        mid = (low + high) // 2
        avg_correct = counterfactual_test(train_idxs[:mid], flip_class, eval_idx, num_tests=num_tests,
                                          arch=arch)
        if avg_correct > 0.5:
            low = mid
        else:
            high = mid
            min_samples = min(mid, min_samples) # update min_samples if successful in flipping
        print(f"Sample Idx: {eval_idx}, Min samples: {min_samples}")
    return min_samples # return the minimum number of samples to remove

