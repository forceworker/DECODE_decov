import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .utils import *

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DANN(object):
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def DANN_model(self, celltype_num):
        feature_num = len(self.used_features)
        
        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, 512, 0), 
                                        EncoderBlock(512, 256, 0.3))

        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2), 
                                          nn.Linear(128, celltype_num), 
                                          nn.Softmax(dim=1))
        
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2), 
                                              nn.Linear(128, 1), 
                                              nn.Sigmoid())

        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        model_da.append(self.discriminator_da)
        return model_da

    def prepare_dataloader(self, source_data, target_data, batch_size):
        ### Prepare data loader for training ###
        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        target_ratios = [target_data.obs[ctype] for ctype in self.labels]
        self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def train(self, source_data, target_data):

        ### prepare model structure ###
        self.prepare_dataloader(source_data, target_data, self.batch_size)
        self.model_da = self.DANN_model(self.celltype_num).cuda()

        ### setup optimizer ###
        optimizer_da1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.predictor_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)
        optimizer_da2 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)
        
        criterion_da = nn.BCELoss().cuda()
        source_label = torch.ones(self.batch_size).unsqueeze(1).cuda()   # 定义source domain label为1
        target_label = torch.zeros(self.batch_size).unsqueeze(1).cuda()  # 定义target domain label为0
        
        metric_logger = defaultdict(list) 
        loss1 = []
        loss2 = []
        loss3 = []
        for epoch in range(self.num_epochs):
            self.model_da.train()

            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                embedding_source = self.encoder_da(source_x.cuda())
                embedding_target = self.encoder_da(target_x.cuda())
                frac_pred = self.predictor_da(embedding_source)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # caculate loss 
                pred_loss = L1_loss(frac_pred, source_y.cuda())       
                pred_loss_epoch += pred_loss.data.item()
                disc_loss = criterion_da(domain_pred_source, source_label[0:domain_pred_source.shape[0],]) + criterion_da(domain_pred_target, target_label[0:domain_pred_target.shape[0],])
                disc_loss_epoch += disc_loss.data.item()
                loss = pred_loss + disc_loss

                # update weights
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()

                embedding_source = self.encoder_da(source_x.cuda())
                embedding_target = self.encoder_da(target_x.cuda())
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # caculate loss 
                disc_loss_DA = criterion_da(domain_pred_target, source_label[0:domain_pred_target.shape[0],]) + criterion_da(domain_pred_source, target_label[0:domain_pred_source.shape[0],]) 
                disc_loss_DA_epoch += disc_loss_DA.data.item()

                # update weights
                optimizer_da2.zero_grad()
                disc_loss_DA.backward(retain_graph=True)
                optimizer_da2.step()

            pred_loss_epoch = pred_loss_epoch/(batch_idx + 1)
            loss1.append(pred_loss_epoch)
            disc_loss_epoch = disc_loss_epoch/(batch_idx + 1)
            loss2.append(disc_loss_epoch)
            disc_loss_DA_epoch = disc_loss_DA_epoch/(batch_idx + 1)
            loss3.append(disc_loss_DA_epoch)
        
            if (epoch+1) % 1 == 0:
                print('============= Epoch {:02d}/{:02d} in stage3 ============='.format(epoch + 1, self.num_epochs))
                print("pred_loss=%f, disc_loss=%f, disc_loss_DA=%f" % (pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch))
        return loss1, loss2, loss3
    
    def prediction(self):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            logits = self.predictor_da(self.encoder_da(x.cuda())).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth
    