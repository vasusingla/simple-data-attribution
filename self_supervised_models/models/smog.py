
import copy
import lightly
import torch
import torch.nn as nn
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule
from sklearn.cluster import KMeans

class SMoGModel(BenchmarkModule):

    def __init__(self, dataloader_kNN, num_classes,
                lr_factor=0.1, max_epochs=200):
        super().__init__(dataloader_kNN, num_classes)

        self.lr_factor = lr_factor
        self.max_epochs = max_epochs
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a model based on ResNet
        self.projection_head = heads.SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = heads.SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # smog
        self.n_groups = 300
        memory_bank_size = 10000
        self.memory_bank = lightly.loss.memory_bank.MemoryBankModule(size=memory_bank_size)
        # create our loss
        group_features = torch.nn.functional.normalize(
            torch.rand(self.n_groups, 128), dim=1
        )
        self.smog = heads.SMoGPrototypes(group_features=group_features, beta=0.99)
        self.criterion = nn.CrossEntropyLoss()

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def _reset_group_features(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = self.memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def _reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def training_step(self, batch, batch_idx):

        if self.global_step > 0 and self.global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            self._reset_group_features()
            self._reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
            utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        (x0, x1), _, _ = batch

        if batch_idx % 2:
            # swap batches every second iteration
            x0, x1 = x1, x0

        x0_features = self.backbone(x0).flatten(start_dim=1)
        x0_encoded = self.projection_head(x0_features)
        x0_predicted = self.prediction_head(x0_encoded)
        x1_features = self.backbone_momentum(x1).flatten(start_dim=1)
        x1_encoded = self.projection_head_momentum(x1_features)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(x1_encoded)
        group_features = self.smog.get_updated_group_features(x0_encoded)
        logits = self.smog(x0_predicted, group_features, temperature=0.1)
        self.smog.set_group_features(group_features)

        loss = self.criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        self.memory_bank(x0_encoded, update=True)

        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters()) + list(self.prediction_head.parameters())        
        optim = torch.optim.SGD(
            params, 
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

