import copy
import lightly
import torch
import torch.nn as nn
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule

class SimSiamModel(BenchmarkModule):
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
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.ProjectionHead([
            (
                512,
                2048,
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            ),
            (
                2048,
                2048,
                nn.BatchNorm1d(2048),
                None
            )
        ])
        self.criterion = lightly.loss.NegativeCosineSimilarity()
            
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2, # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
