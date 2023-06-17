
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12, att_drop_prob=0.1, state_drop_prob=0.5, return_att_score=False,device="cuda:0"):
        super().__init__()
        self.dim = d_model
        self.num_heads = num_heads
        self.att_drop_prob = att_drop_prob
        self.state_drop_prob = state_drop_prob
        self.return_att_score=return_att_score
        self.device = device
        self.size_per_head = self.dim // self.num_heads  # 64
        self.Wq = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wk = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.W = nn.Linear(self.num_heads * self.size_per_head, self.dim)
        self.lm = nn.LayerNorm(self.dim)
        self.att_drop = nn.Dropout(self.att_drop_prob)
        self.state_drop = nn.Dropout(self.state_drop_prob)

    def calc_mask_score(self, attention_mask, aim_shape):
        """
         * @description: 计算mask_score
         * @param  self : 
         * @param  attention_mask : (B,S)
         * @param  aim_shape : (B,H_num,S,H_dim)
         * @return 
        """
        mask_score = torch.zeros(size=aim_shape).to(self.device)
        print(mask_score.shape, attention_mask[:, None, None, :].shape)
        mask_score = mask_score + attention_mask[:, None, None, :]
        mask_score = (1.0 - mask_score) * -1000000.0
        return mask_score

    def SelfAttention(self, q, k, v, attention_mask):
        """
         * @description: 注意力加残差
         * @param  self : 
         * @param  q :  bxLxd
         * @param  k : bxSxd
         * @param  v : bxsxd
         * @param  attention_mask :          attention_mask: # bxS
                                                    1 normal token
                                                    0 masked token
         * @return  bxLxd
        """
        Q_new_size = q.size()[:-1] + (self.num_heads, self.size_per_head)  # b, L, h, h_dim
        K_new_size = k.size()[:-1] + (self.num_heads, self.size_per_head)  # b, S, h, h_dim
        Q = self.Wq(q).view(*Q_new_size).permute(0, 2, 1, 3)  ## b ,H , L,h_dim
        K = self.Wk(k).view(*K_new_size).permute(0, 2, 1, 3)  ## b ,H , S,h_dim
        V = self.Wv(v).view(*K_new_size).permute(0, 2, 1, 3)  ## b ,H , S,h_dim
        attention_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.size_per_head)  ## b ,H,L ,S
        # attention mask here
        attention_score = attention_score + self.calc_mask_score(attention_mask, attention_score.shape)
        attention_score = nn.Softmax(dim=-1)(attention_score)
        attention_score = self.att_drop(attention_score)
        O = torch.matmul(attention_score, V)
        O = self.W(O.permute(0, 2, 1, 3).contiguous().view(q.size(0), q.size(1), -1))  # bxLxd
        O = self.state_drop(O)
        O = self.lm(q + O)
        mean_head_att_score= attention_score.mean(dim=1)
        return O,mean_head_att_score

    def forward(self, q, k, v, attention_mask):
        """
         * @description: 注意力加残差
         * @param  self : 
         * @param  q : bxLxd
         * @param  k : bxsxd
         * @param  v : bxsxd
         * @param  attention_mask : # bxS
                    1 normal token
                    0 masked token
         * @return xb :xLxd ; att_score: B * L * S
        """
        x,att_score = self.SelfAttention(q, k, v, attention_mask)
        if self.return_att_score:
            return x,att_score
        else:
            return x
            
