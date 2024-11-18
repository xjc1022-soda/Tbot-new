import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from .transformer import TbotEncoder
from .transformer import TbotDecoder
from .transformer import MultiHeadAttention
from .transformer import Merger_pool
from .heads import PretrainHead, PredictionHead
import torch.fft as fft
from augmentations import DataTransform_TD_bank

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

def visible_mask_div(x, mask_ratio):
    """
    Divide the input tensor into visible and mask parts.
    x: tensor [bs x n_patch x n_vars x patch_len]
    """
    n_mask = int(x.size(1) * mask_ratio)
    total_idx = torch.arange(x.size(1))
    
    # Generate a random permutation of indices and select the first n_mask indices
    perm = torch.randperm(x.size(1))
    mask_idx = perm[:n_mask]
    
    # Create a boolean mask
    bool_mask = torch.zeros(x.size(1), dtype=torch.bool)
    bool_mask[mask_idx] = True
    
    # Let visible_idx be the complement of mask_idx
    visible_idx = total_idx[~bool_mask]
    
    visible_ts = x[:, visible_idx, :, :]
    mask_ts = x[:, mask_idx, :]
    return visible_ts, mask_ts

class Tbot(nn.Module):
    """
    Use PatchTST (Transfomer) or TS2Vec (dilated CNN) as encoder.
    This class will be used to train the model in main.py.
    """
    
    def __init__(
        self,
        n_vars,
        num_patch,
        forecast_len,
        head_type,
        batch_size,
        dropout=0,
        mask_ratio=0.2,
        d_model=64,
        n_hierarchy=3,
        n_layers=2, 
        d_k=64, 
        d_v=64, 
        d_ff=64, 
        n_heads=4,
        patch_len=24,
        stride=24,
        individual=False,
        alpha=0.5,
        beta=0.5,
        device='cuda',
        lr=0.001,
        pe='zeros', 
        learn_pe=True, 
        after_epoch_callback=None
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_len
        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.num_patch = num_patch
        self.n_hierarchy = n_hierarchy
        
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification']

        # Student and teacher encodernetwork
        # num_vis = num_patch - int(num_patch * mask_ratio)
        self.student_net_t = TbotEncoder(n_vars, patch_len, d_model, n_hierarchy, n_layers, 
                            d_k, d_v, d_ff, n_heads, pe, learn_pe, num_patch, True, dropout).to(self.device)
        self.teacher_net_t = copy.deepcopy(self.student_net_t).to(self.device)
        self.student_net_f = TbotEncoder(n_vars, patch_len, d_model, n_hierarchy, n_layers, 
                            d_k, d_v, d_ff, n_heads, pe, learn_pe, num_patch, True, dropout).to(self.device)
        self.teacher_net_f = copy.deepcopy(self.student_net_f).to(self.device)        

        for param in self.teacher_net_t.parameters():
            param.requires_grad = False
        for param in self.teacher_net_f.parameters():
            param.requires_grad = False

        # Decoder network and mask query
        self.decoder_t = TbotDecoder(patch_len , d_model, n_hierarchy, d_k, d_v, d_ff, n_heads).to(self.device)
        self.decoder_f = TbotDecoder(patch_len , d_model, n_hierarchy, d_k, d_v, d_ff, n_heads).to(self.device)
        self.t_query = torch.randn(batch_size * n_vars, int(self.num_patch * mask_ratio) ,d_model).to(self.device)
        self.f_query = torch.randn(batch_size * n_vars, int(self.num_patch * mask_ratio) ,d_model).to(self.device)
        self.f_query.requires_grad = True
        self.t_query.requires_grad = True

        self.merger = Merger_pool().to(self.device)

        # Head

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, num_patch, n_hierarchy, d_k, d_v, n_heads, dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, n_vars, d_model, d_k, d_v, n_heads, num_patch, n_hierarchy, forecast_len)

        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0

        self.optimizer = torch.optim.Adam([*self.student_net.parameters(), self.t_query], lr=self.lr)
    
    def forward(self, z):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        # print(z.shape)
        outputs = self.student_net_t(z)               # outputs: [bs x num_patch/2 ** i x n_vars x d_model]
        z = torch.cat(outputs, dim=1)               # z: [bs x 7/4 *num_patch x nvars x d_model]
        z = self.head(z)                                                                    
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z
    
    def reconstruct(self, x, f):
        x_vis, x_mask = visible_mask_div(x, self.mask_ratio)
        f_vis, f_mask = visible_mask_div(f, self.mask_ratio)
        outputs = self.student_net(x_vis)
        outputs_f = self.student_net(f_vis)
        _, pred_mask_1 = self.decoder_t(self.t_query, outputs)
        _, pred_mask_2 = self.decoder_t(self.t_query, outputs_f)
        _, pred_mask_f_1 = self.decoder_f(self.f_query, outputs)
        _, pred_mask_f_2 = self.decoder_f(self.f_query, outputs_f)
        r_loss_t = 0
        r_loss_f = 0
        for i in range(self.n_hierarchy):
            r_loss_t += F.mse_loss(pred_mask_1[i], x_mask) + F.mse_loss(pred_mask_2[i], x_mask)
            r_loss_f += F.mse_loss(pred_mask_f_1[i], f_mask) + F.mse_loss(pred_mask_f_2[i], f_mask)
            bs, _, n_vars, d_model  = x_mask.shape
            x_mask = torch.reshape(x_mask, (bs, -1, n_vars * d_model))
            x_mask = self.merger(x_mask).reshape(bs, -1, n_vars, d_model)
            f_mask = torch.reshape(f_mask, (bs, -1, n_vars * d_model))
            f_mask = self.merger(f_mask).reshape(bs, -1, n_vars, d_model)
        return (r_loss_t + r_loss_f) / 4
    
    def distill(self, x, f):
        config = augmentations()
        x_1, x_2 = x, DataTransform_TD_bank(x, config)
        f_1 = fft.fft(x_1)
        f_2 = fft.fft(x_2)
        t_1_out = self.student_net_t(x_1)
        f_1_out = self.student_net_f(f_1)
        t_2_out = self.student_net_t(x_2)
        f_2_out = self.student_net_f(f_2)
        t_t_loss = 0
        t_f_loss = 0
        f_f_loss = 0
        for i in range(self.n_hierarchy):
            t_t_loss += F.mse_loss(t_1_out[i], t_2_out[i])
            t_f_loss += F.mse_loss(t_1_out[i], f_2_out[i]) + F.mse_loss(t_2_out[i], f_1_out[i])
            f_f_loss += F.mse_loss(f_1_out[i], f_2_out[i])
        loss = (t_t_loss + t_f_loss + f_f_loss) / 4
        return loss

    def cal_Loss(self, x, f):
        # print(x.shape, f.shape)
        r_loss = self.reconstruct(x, f)
        d_loss = self.distill(x, f)
        loss = self.alpha * d_loss + self.beta * r_loss
        return loss
    
    
        # x_vis, x_mask = visible_mask_div(x, self.mask_ratio)
        # outputs = self.student_net(x_vis)
        # z_list, pred_list = self.decoder(self.t_query, outputs)
        # z_hat_list = self.teacher_net(x_mask)
        # d_loss = 0
        # r_loss = 0
        # for i in range(self.n_hierarchy):
        #     # print(outputs[i].shape, z_list[i].shape, z_hat_list[i].shape, pred_list[i].shape)
        #     d_loss += F.mse_loss(z_list[i], z_hat_list[i])
        #     r_loss += F.mse_loss(pred_list[i], x_mask)
        #     bs, _, n_vars, d_model  = x_mask.shape
        #     x_mask = torch.reshape(x_mask, (bs, -1, n_vars * d_model))
        #     x_mask = self.merger(x_mask).reshape(bs, -1, n_vars, d_model)
        # loss = self.alpha * d_loss + self.beta * r_loss



    def ema_update(self, alpha=0.99):
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = alpha * param_t.data + (1 - alpha) * param_s.data

    # def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
    #     ''' Training the TiBot model.
        
    #     Args:
    #         train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
    #         n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
    #         n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
    #         verbose (bool): Whether to print the training loss after each epoch.
            
    #     Returns:
    #         loss_log: a list containing the training losses on each epoch.
    #     '''
    #     assert train_data.ndim == 3

    #     if n_epochs is None:
    #         n_epochs = 400
    #     loss_log = []

    #     train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
    #     train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        
    #     for _ in range(n_epochs):
    #         for i, (batch,) in enumerate(train_loader):
    #             # self.teacher_net detach

    #             self.optimizer.zero_grad()
                
    #             # copy the student weights to the teacher 
    #             # self.teacher.load_state_dict(self.student.state_dict())

    #             x = batch.to(self.device)
    #             # print(x.shape)
    #             # print(f'x shape: {x.shape}')
    #             x_patch = self.patch(x)
    #             # print(x_patch.shape)
    #             visible_patch, mask_patch = visible_mask_div(x_patch, self.mask_ratio)
    #             # print(mask_patch.shape)
    #             # print(visible_patch.shape)
    #             # print(self.n_patch)
    #             assert visible_patch.shape[1] + mask_patch.shape[1] == self.n_patch              
    #             outputs, attens = self.student_net(visible_patch, )
    #             z, pred = self.decoder(self.t_query, outputs)
    #             z_hat, _ = self.teacher_net(mask_patch)

    #             # for i in range(3):
    #             #     print(outputs[i].shape, z[i].shape, z_hat[i].shape, pred[i].shape)
 
    #             d_loss = DistillationLoss(z, z_hat)
    #             r_loss = ReconstructionLoss(mask_patch, pred, self.merger)
    #             loss = self.alpha * d_loss + self.beta * r_loss
    #             loss.mean().backward()
    #             self.optimizer.step()
    #             self.teacher_net = copy.deepcopy(self.student_net).to(self.device)
    #             # self.ema_update()
    #             # import ipdb
    #             # ipdb.set_trace()
            
    #         print(f'Epoch {self.n_epochs+1} loss: {loss.mean().item()}')
    #         self.n_epochs += 1

    #     return loss
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    



        # self.patch = Patching(patch_size, stride, padding).to(self.device)  
        # self.patch_num = [int(n_patch / 2**i) for i in range(n_hierarchy)]