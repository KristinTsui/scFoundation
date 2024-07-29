import sys 
import numpy as np
import torch
from torch import nn
sys.path.append("../model/") # path to this folder
from load import *

class FinetunePatientClassification(nn.Module):

    def __init__(self, ckpt_path, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self): # confgure the model
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False

            for na, param in self.encoder.named_parameters():
                param.requires_grad = False
            for na, param in self.encoder.transformer_encoder[-2].named_parameters():
                print('self.encoder.transformer_encoder ', na, ' have grad')
                param.requires_grad = True

        self.fc1 = nn.Sequential(
            nn.Linear(model_config['encoder']['hidden_dim'], 256), # number of input features is the hidden dimension of the encoder
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6) # add a batch normalization layer to the model
        self.model_config = model_config

        # ensure the final layer requires gradients
        for na, param in self.fc1.named_parameters():
            print('self.fc1 ', na, ' have grad')
            param.requires_grad = True

    def forward(self, sample_list, *arg, **kwargs):
        

        targets = sample_list['targets'] # get the target labels from sample_list

        x = sample_list['x'] # get the expression data from the current batch (B, L)
        value_labels = x > 0 # create a boolean mask where expression is non-zero
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id']) # pad the zeros in the expression
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1) # generate gene ids for expression data
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, 
                                          self.model_config['pad_token_id']) # pad the position ids 
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0) # pass the input data through the token embedding layer
       
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb # add the position embedding to token embedding


        logits = self.encoder(x, x_padding) # pass the encoded data through the encoder

        # mlp
        logits, _ = torch.max(logits, dim=1) # get the maximum value of the logits along the first dimension

        logits = self.norm(logits) # normalize the logits
        logits = self.fc1(logits) # pass the logits through the linear layer


        return logits
    
    def compute_loss(self, logits, targets):
        # log the shape of the logits and targets
        # print('logits shape: ', logits.shape)
        # print('target shape: ', targets.shape)

        # Squeeze logits to match target shape
        if logits.dim() == 2:
            logits = logits.squeeze(1) # remove the second dimension of the logits
        
        # print('squeezed logits shape: ', logits.shape)

        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    
if __name__ == '__main__':
    finetune_model = FinetunePatientClassification(ckpt_path='./models/models.ckpt')
    sample_list = {'x': torch.zeros([8, 18264]).cuda(), 'targets': torch.randint(0, 2, (8,)).float().cuda()} # create some dummy data to test the model
    sample_list['x'][:,:100] = 1 # set the first 100 values of x to 1
    finetune_model.build() # build the model
    finetune_model = finetune_model.cuda()
    logits = finetune_model(sample_list) # get the logits from the model
    
