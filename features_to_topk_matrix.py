import os
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import svm
from tqdm import tqdm

model_dir = '' # Add model directory here
required_test_indices_path = './data/test_indices/test_100.npy'
save_topk_train_samples_dir = './data/topk_train_samples/'
CIFAR10_ROOT = os.getenv('CIFAR10_ROOT')

def main():
    # Arguments to modify
    reg_C = 0.1                  # SVM regularization parameter
    topk = 1280                  # Top k training samples to save
    model_name = 'Moco_800epoch' 
    label_conditioning = True
    distance_key = 'esvm' # 'esvm' for exemplar svm or 'l2knn' for euclidean distance KNN

    # No need to modify strings below
    run_name = f'tmp_requiredidx_{model_name}_{distance_key}_v2' + (f'_c{str(reg_C).replace(".", "")}' if distance_key in ['esvm', 'svm'] else '')
    conditioning_str = 'class_conditioned' if label_conditioning else 'unconditioned'
    print(f"Model name: {model_name}")
    print(f"Label conditioning: {label_conditioning}")
    print(f"Distance key: {distance_key}")
    print(f"Run name: {run_name}")

    feature_dir = os.path.join(model_dir, model_name)
    train_features = np.load(os.path.join(feature_dir, 'train.npy'))
    test_features = np.load(os.path.join(feature_dir, 'test.npy'))

    cifar_train_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True)
    cifar_test_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False)
    train_labels = np.array([cifar_train_ds[i][1] for i in range(len(cifar_train_ds))])
    test_labels = np.array([cifar_test_ds[i][1] for i in range(len(cifar_test_ds))])

    print(f"test features have the following shape: {test_features.shape}")
    print(f"test labels have the following shape: {test_labels.shape}")


    target_to_nearest_train_idx = np.zeros((10000, topk), dtype=np.int64)
    required_test_indices = np.load(required_test_indices_path)
    
    # For every target sample, we save topk indices of the topk training samples
    for i, cur_feature in tqdm(enumerate(test_features), desc='Computing topk matrix'):
            if i not in required_test_indices:
                continue
            
            # if label conditioning, we select training samples with same label as target
            if label_conditioning:
                cur_label = test_labels[i]
                train_indices_of_same_class = np.where(train_labels == cur_label)[0]
                cur_train_features = train_features[train_indices_of_same_class]
            else:
                cur_train_features = train_features
            
            # For ffcv_resnet package, every train image must be done one at a time
            # add back first dimension to cur feature
            cur_feature = np.expand_dims(cur_feature, axis=0)
            distances = get_distance(distance_key, cur_feature, cur_train_features, reg_C)

            if label_conditioning:
                topk_indices = train_indices_of_same_class[np.argsort(distances)[:topk]]
            else:
                topk_indices = np.argsort(distances)[:topk]

            target_to_nearest_train_idx[i] = topk_indices

    print('Created target_to_nearest_train_idx matrix with shape', target_to_nearest_train_idx.shape)
    
    # Save the topk matrix
    save_dir = os.path.join(save_topk_train_samples_dir, conditioning_str)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{run_name}_{topk}.npy'), target_to_nearest_train_idx)

    cifar_train_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True)
    cifar_test_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False)

    target_idx = 6218

    # plot target image followed by top 49 nearest training images
    # should not override the target image
    fig, axs = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle(run_name, fontsize=16)
    axs[0, 0].imshow(cifar_test_ds[target_idx][0])
    axs[0, 0].set_title(f'Target, Class {cifar_test_ds[target_idx][1]}')
    axs[0, 0].axis('off')

    for i in range(49):
        row_idx = i // 10
        column_idx = (i % 10) + 1
        if column_idx == 10:
            column_idx = 0
            row_idx += 1
        
        axs[row_idx, column_idx].imshow(cifar_train_ds[target_to_nearest_train_idx[target_idx][i]][0])
        axs[row_idx, column_idx].set_title(f'#{i+1}, Class {cifar_train_ds[target_to_nearest_train_idx[target_idx][i]][1]}')
        axs[row_idx, column_idx].axis('off')

    # Save the plot
    print(f'Saving plot to target{target_idx}_{run_name}_{conditioning_str}.png')
    fig.savefig(f'target{target_idx}_{run_name}_{conditioning_str}', dpi=300)
    
def get_distance(distance_key, cur_feature, cur_train_features, reg_C):
    if distance_key == 'esvm':
        return get_exemplar_svm_distance(cur_feature, cur_train_features, reg_C)
    elif distance_key == 'l2knn':
        euclidean_distances = torch.cdist(torch.from_numpy(cur_feature), torch.from_numpy(cur_train_features), p=2)
        euclidean_distances = euclidean_distances[0]
        return euclidean_distances.numpy() 
    else:
        raise ValueError(f"Invalid distance key: {distance_key}")

def get_exemplar_svm_distance(test_feature, train_features, reg_C):
    """
    returns svm distance of test_feature to train_features
    """
    svm_x = np.concatenate((test_feature, train_features), axis=0)
    svm_y = np.zeros(svm_x.shape[0])
    svm_y[0] = 1 # the target feature is our only positive example

    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=reg_C)
    clf.fit(svm_x, svm_y)

    # get the top features
    similarities = clf.decision_function(train_features)
    return -similarities


if __name__ == "__main__":
    main()