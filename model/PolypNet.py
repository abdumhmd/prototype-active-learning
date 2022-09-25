import torchvision.models as models
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import Config

class PolypNet(pl.LightningModule):
    def __init__(self, num_classes=1,lr=Config.lr):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet50(pretrained=True)
        num_filters = self.model.fc.in_features
        
        for param in self.model.parameters():
            param.requires_grad=False

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.model.fc = nn.Linear(num_filters, num_classes)


        

    def forward(self, input_data):
        preds = self.model(input_data)
        # preds=nn.Sigmoid()(preds)
        return preds.squeeze(1)

    def training_step(self,batch,batch_idx):
        x,y=batch
        preds=self.forward(x)
        loss=F.binary_cross_entropy_with_logits(preds,y.float())

        
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('Training Accuracy', Accuracy()(preds,y))
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        preds=self.forward(x)
        loss=F.binary_cross_entropy_with_logits(preds,y.float())
 
        self.log('validation_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('Validation Accuracy', Accuracy()(preds,y))


    def configure_optimizers(self):
        # return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer