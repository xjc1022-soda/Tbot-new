import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from .transformer import MultiHeadAttention



class PretrainHead(nn.Module):  
    def __init__(self, d_model, patch_len, num_patch, n_hierarchy, d_k, d_v, n_heads, dropout):
        super().__init__()
        self.cross_scale_layer = CrossScaleHead(d_model, num_patch, n_hierarchy, d_k, d_v, n_heads)
        self.dropout = nn.Dropout(dropout)  
        self.linear = nn.Linear(d_model, patch_len)
    
    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x 7/4 * num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        x = self.cross_scale_layer(x)           # [bs x nvars x d_model x num_patch]
        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x       


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, d_k, d_v, n_heads, num_patch, n_hierarchy, forecast_len, head_dropout=0, flatten=False):
        super().__init__()
        self.cross_scale_layer = CrossScaleHead(d_model, num_patch, n_hierarchy, d_k, d_v, n_heads)
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        x = self.cross_scale_layer(x)   # [bs x nvars x d_model x num_patch]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]

class CrossScaleHead(nn.Module):
    def __init__(self, d_model, num_patch, n_hierarchy, d_k, d_v, n_heads):
        super().__init__()
        self.cross_scale_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.shape_list = [num_patch // 2**i for i in range(n_hierarchy)]
        self.n_hierarchy = n_hierarchy
        # let idx_list be the accumulated sum of shape_list
        self.idx_list = [0] + [sum(self.shape_list[:i+1]) for i in range(n_hierarchy)]
        self.linear_sep = nn.ModuleList(nn.Linear(self.shape_list[i], num_patch) for i in range(n_hierarchy))
    def forward(self, x):
        '''
        x: [bs x 7/4 *num_patch x nvars x d_model]
        output: [bs x  nvars x d_model x num_patch]
        '''
        bs, _ ,n_vars, d_model = x.shape
        assert x.shape[1] == sum(self.shape_list)
        x = x.reshape(bs * n_vars, -1, d_model)  # [bs*nvars x 7/4*num_patch x d_model]
        x, _ = self.cross_scale_attn(x, x, x, None)   # [bs*nvars x 7/4*num_patch x d_model]
        x = x.reshape(bs, n_vars, d_model, -1)    # [bs x nvars x d_model x 7/4 * num_patch]
        rep = []
        for i in range(self.n_hierarchy):
            z = self.linear_sep[i](x[:, :, :, self.idx_list[i]:self.idx_list[i+1]])   # [bs x nvars x d_model x num_patch]
            rep.append(z)
        z = torch.sum(torch.stack(rep, dim=0), dim=0)   # [bs x nvars x d_model x num_patch]

        return z


      
# class PreHead(nn.Module):
#     def __init__(self, 
#                  TranBot, 
#                  pred_len,
#                  shape_list, 
#                  ts_size, 
#                  ts_dim,
#                  d_model, 
#                  d_k, 
#                  d_v, 
#                  n_heads,
#                  batch_size,
#                  device='cuda',
#                  lr=0.001,
#                  after_epoch_callback=None):
#         super(PreHead, self).__init__()
#         self.device = device   
#         self.lr = lr
#         self.batch_size = batch_size
#         self.TranBot = TranBot
#         self.pred_len = pred_len
#         self.shape_list = shape_list
#         self.n_epochs = 0
#         for param in self.TranBot.parameters():
#             param.requires_grad = False
#         self.linear_list = nn.ModuleList()
#         self.cross_scale_layer = MultiHeadAttention(d_model, d_k, d_v, n_heads).to(self.device)
#         for shape in shape_list:
#             self.linear_list.append(nn.Linear(shape, pred_len))
#         self.projection = nn.Linear(d_model, ts_dim)
#         self.after_epoch_callback = after_epoch_callback
#         print(self.parameters())
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        

    # def forward(self, x):
    #     # Ensure x is a NumPy array
    #     if isinstance(x, torch.Tensor):
    #         x = x.cpu().numpy()
    #     x = torch.from_numpy(x).to(torch.float).to(self.device)
    #     outputs = self.TranBot.forward(x)
    #     # print(outputs[0].shape)
    #     shape, idx_list = cal_shape_idx(outputs)
    #     # print(f'idx_list: {idx_list}')
    #     outputs_joint = torch.cat(outputs, dim=1)
    #     rep_joint , _ = self.cross_scale_layer(outputs_joint, outputs_joint, outputs_joint, None)
    #     all_rep = []
    #     for i in range(len(shape)):
    #         linear = self.linear_list[i]
    #         # print(idx_list[i], idx_list[i+1])
    #         rep = rep_joint[:, idx_list[i]:idx_list[i+1],:]
    #         # print(f'rep.shape: {rep.shape}')
    #         # print(shape[i])
    #         rep = linear(rep.transpose(1, 2)).transpose(1, 2)
    #         all_rep.append(rep)
    #     all_rep = torch.sum(torch.stack(all_rep, dim=0), dim=0)

    #     return all_rep
    
    # def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
    #     assert train_data.ndim == 3

    #     if n_epochs is None:
    #         n_epochs = 400
    #     loss_log = []

    #     train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
    #     train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)        


    #     for _ in range(n_epochs):
    #         for i, (batch,) in enumerate(train_loader):
    #             self.optimizer.zero_grad()
    #             x = batch.to(self.device)
    #             x = x[:, :-self.pred_len,:]
    #             gt = x[:, -self.pred_len:,:]
    #             all_rep = self.forward(x)
    #             pred = self.projection(all_rep)
    #             loss = F.mse_loss(pred, gt)
    #             loss.mean().backward()
    #             self.optimizer.step()

            
    #         print(f'Epoch {self.n_epochs+1} loss: {loss.mean().item()}')
    #         self.n_epochs += 1

    #     return loss