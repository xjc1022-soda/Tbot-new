import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_head, len_q, d_k]
        K: [batch_size, n_head, len_k, d_k]
        V: [batch_size, n_head, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_head, seq_len, seq_len]
        '''
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_head, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_head, len_q, d_v]
        return torch.matmul(attn, V), attn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # print(batch_size)
        residual, batch_size = input_Q, input_Q.size(0)
        # print(batch_size)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1,2)

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)  
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
    
class EncoderHier(nn.Module):
    def __init__(self, n_layers, d_model, d_k, d_v, d_ff, n_head):
        super(EncoderHier, self).__init__()
        self.n_layers = n_layers
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [bs * n_vars x num_patch x d_model]
        enc_self_attn_mask: None
        '''
        for _ in range(self.n_layers):
            enc_outputs, _ = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_outputs: [bs * n_vars x num_patch x d_model]
            enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [bs * n_vars x num_patch x d_model]
            enc_inputs = enc_outputs
        return enc_outputs


class Merger_pool(nn.Module):
    def __init__(self):
        super(Merger_pool, self).__init__()
        self.pool = nn.AvgPool1d(2, padding=0)
    def forward(self, enc_outputs):
        # print(enc_outputs.shape)
        return self.pool(enc_outputs.transpose(1, 2)).transpose(1, 2)



class TbotEncoder(nn.Module):
    def __init__(self, c_in, patch_len, d_model, n_hierarchy, n_layers, d_k, d_v, d_ff, n_heads, pe, learn_pe, num_patch, shared_embedding=True, dropout=0):
        super(TbotEncoder, self).__init__()
        self.shared_embedding = shared_embedding     
        self.d_model = d_model
        if not shared_embedding: 
            self.value_embedding = nn.ModuleList()
            for _ in range(self.n_vars): 
                self.value_embedding.append(nn.Linear(patch_len, d_model))
        else:
            self.value_embedding = nn.Linear(patch_len, d_model) 
        # print(f'pe: {pe}, learn_pe: {learn_pe}')
        self.position_embedding = positional_encoding(pe, learn_pe, num_patch, d_model)
        self.dropout = nn.Dropout(dropout)
        self.hiers = nn.ModuleList([EncoderHier(n_layers, d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_hierarchy)])
        self.merger = Merger_pool()
    def forward(self, x): 
        '''
        x: [bs x num_patch x n_vars x patch_len]
        output[0]: [bs x nvars x d_model x num_patch]
        '''
        bs, num_patch, n_vars, patch_len = x.shape
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.value_embedding[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.value_embedding(x)                 # x: [bs x num_patch x n_vars x d_model]
        x = x.transpose(1,2)
        # print(f'x_shape is {x.shape}')                            # x: [bs x  n_vars x num_patch x d_model]
        u = torch.reshape(x, (bs * n_vars, num_patch, -1))    # u: [bs * n_vars x num_patch x d_model]
        # print(u.shape, self.position_embedding.shape)
        u = self.dropout(u + self.position_embedding[:num_patch,:])   #u : [bs * n_vars x num_patch x d_model]
    
        enc_outputs_list = []
        for hier in self.hiers:
            z = hier(u, None)                       # z: [bs * n_vars x num_patch x d_model]
            u = self.merger(z)                      # u: [bs * n_vars x num_patch/2 x d_model]
            output = torch.reshape(z, (bs, -1, n_vars, self.d_model))   # output: [bs x num_patch/2 ** i x n_vars x d_model]
            enc_outputs_list.append(output)
        return enc_outputs_list

class TbotDecoder(nn.Module):
    def __init__(self, patch_len, d_model, n_hierarchy, d_k, d_v, d_ff, n_head):
        super(TbotDecoder, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.dec_cross_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.project = nn.Linear(d_model, patch_len)
        self.merger = Merger_pool()
        self.n_hierarchy = n_hierarchy
        self.d_model = d_model
    def forward(self, m_query, enc_outputs_list): #  enc_outputs_list [bs x num_patch/2 ** i x n_vars x d_model]
        bs, _, n_vars, d_model  = enc_outputs_list[0].shape
        z_list = []
        pred_list = []
        for i in range(self.n_hierarchy):
            output = enc_outputs_list[i].reshape(bs * n_vars, -1, d_model) # [bs * n_vars x num_patch x d_model]
            K, V =  output, output  # K, V: [bs * n_vars x num_patch x d_model]
            z , _ = self.dec_cross_attn(m_query, K, V, None)  # m_query: [bs * n_vars x num_mask x d_model]
            z = self.pos_ffn(z)                        # z: [bs * n_vars  x num_mask x d_model]
            z , _ = self.dec_self_attn(z, z, z, None)  # z: [bs * n_vars  x num_mask x d_model]
            z = self.pos_ffn(z)
            z = z.reshape(bs, -1, n_vars, self.d_model) # z: [bs x n_vars x num_mask x d_model]
            pred = self.project(z)           # pred: [bs x num_mask x n_vars x patch_len]
            z_list.append(z)
            pred_list.append(pred)
            m_query = self.merger(m_query)
        return z_list, pred_list
    

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)



def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe




