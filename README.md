## Simple Efficient Data Attribution

This work contains all the code for the paper **A Simple and Efficient Baseline for Data Attribution on Images** by Vasu Singla, Pedro Sandoval-Segura, Micah Goldblum, Jonas Geiping, Tom Goldstein 


### Installation

This repository requires [FFCV](https://github.com/libffcv/ffcv) library, and [PyTorch](https://pytorch.org/). You can also install the environment via 

```
conda env create -f environment.yml
``` 

### Data 

All the data used for the paper is provided [Google Drive Link](https://drive.google.com/drive/folders/10_WMZ4c8Co_VV-i3isoPcdM-t9q0-VuL?usp=drive_link). We describe all the data included below - 


1. **Top-k Train Samples** - For our repository, we pre-compute the closest top-k training samples from each method and our baselines. These are also provided in the link under the subfolders `cifar10/topk_train_samples` and `imagenet/topk_train_samples` for CIFAR-10 and Imagenet respectively.
2. **Test Indices** - We randomly selected 100 and 30 test samples for CIFAR-10 and Imagenet used throughout the paper, these are provided at `cifar10/test_indices` and `imagenet/test_indices`.
3. **Mislabel Support MetaData** - To compute mislabel support, we also need to specify which class to flip a test sample to. For CIFAR-10, we trained 10 Resnet-9 models for this task, and for Imagenet we trained 4 Resnet-18 models. The average predictions of these are provided in the link above. The metadata also requires labels for the dataset which are included above. 
4. **Models** - *Note that the models used are not required to run this code, only the top-k training samples are required*. However, for transperancy the link also contains our trained Self-Supervised Models, and DataModel Weights for CIFAR-10. All of these are provided at the link [here](https://drive.google.com/drive/folders/1Nh_3lZx_sn0_bANoNJGizfvXfWc5Bmz5?usp=sharing). For Imagenet MoCo model, you directly download it from the official [repo](https://github.com/facebookresearch/moco). For reproducing TRAK, you can follow the tutorial from the author's original [code](https://github.com/MadryLab/trak).

### Counterfactual Estimation on CIFAR-10

To perform counterfactual estimation for a single test sample on CIFAR-10 run the following - 

```
python counterfactual_search.py --test-idx $test_idx \
                                --matrix-path $matrix_path \
                                --results-path $results_path \
                                --num-tests 5 \
                                --search-budget 7 \
                                --arch $arch
```

The arguments are defined as follows - 

```
--test-idx       Specifies the test index on which to perform counterfactual estimation
--matrix-path    Path to matrix containing top-k **training indices for each validation sample**
--results-path   Path where results for the test sample are dumped as a pickle file
--search-budget  Budget to use for bisection search
--arch           Model architecture to use {resnet-9, mobilenetv2}
--flip-class     Boolean argument, if specified computes mislabel support instead of removal support
```

When using `--flip-class`, you also need to specify where the metadata regarding the test labels and second predicted class using `--label-path` and `--rank-path`. This metadata is provided in the data above. 

### CounterFactual Estimation on Imagenet

TODO. This has a few of our SLURM stuff built-in that needs to be removed for release. In the meantime if you want, you can adapt the code we used from [FFCV Imagenet](https://github.com/libffcv/ffcv-imagenet/tree/main) to do counterfactual estimation. 

### Self Supervised Models - CIFAR

To train CIFAR-10 SSL models, use the `self_supervised_models` subfolder. The `train_ssl.py` script provides an interface for the same. 

### Citation 

If you find our code useful, please consider citing our work -

```
TBD
```

If you run into any problems, please raise a Github Issue, we'll be happy to help!

### Acknowledgments 
 
The Datamodels weights on CIFAR-10 using 50% of the data were downloaded from [here](https://github.com/MadryLab/datamodels-data). We also trained our own datamodels using code available [here](https://github.com/MadryLab/datamodels/tree/main).

The TRAK models were trained using code available [here](https://github.com/MadryLab/trak).

FFCV Imagenet training code was used from [here](https://github.com/libffcv/ffcv-imagenet/tree/main). 

The Self-Supervised models were trained using [Lightly Benchmark Code](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).