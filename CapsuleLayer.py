"""
# * @Author: DuTim
# * @Date: 2023-03-09 20:38:59
# * @LastEditTime: 2023-06-19 10:29:15
# * @Description: capsule network to extractor multi-interest 
"""
import torch
import torch.nn as nn

class Routing(nn.Module):
    def __init__(self, emb_size, max_his, iterations, K, relu_layer):
        super().__init__()
        self.emb_size = emb_size
        self.max_his = max_his
        self.iterations = iterations
        self.K = K
        self.S = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.relu_layer = relu_layer
        if self.relu_layer:
            self.layer = nn.Sequential(nn.Linear(self.emb_size, self.emb_size), nn.ReLU())

    @staticmethod
    def squash(x):
        x_squared_len = (x**2).sum(-1, keepdim=True)
        scalar_factor = x_squared_len / (1 + x_squared_len) / torch.sqrt(x_squared_len + 1e-9)
        return x * scalar_factor

    def forward(self, low_capsule, valid_his):
        # low_capsule : [batch_size, seq_len, emb_size] ; valid_his: bsz,seq_len(0代表mask)
        batch_size, seq_len, _ = low_capsule.shape
        B = nn.init.normal_(torch.empty(batch_size, self.K, seq_len), mean=0.0, std=1.0).to(low_capsule.device)
        low_capsule_new = self.S(low_capsule)
        low_capsule_new = low_capsule_new.repeat(1, 1, self.K).reshape((-1, seq_len, self.K, self.emb_size))
        low_capsule_new = low_capsule_new.transpose(1, 2)  # [batch_size, K, seq_len, emb_size]
        low_capsule_iter = low_capsule_new.detach()
        for i in range(self.iterations):
            atten_mask = valid_his[:, None, :].repeat(1, self.K, 1)
            paddings = torch.zeros_like(atten_mask).float()
            W = B.softmax(1)  # [batch_size, K, seq_len]
            W = torch.where(atten_mask == 0, paddings, W)
            W = W[:, :, None, :]  # [batch_size, K, 1, seq_len]
            if i + 1 < self.iterations:
                Z = torch.matmul(W, low_capsule_iter)  # [batch_size, K, 1, emb_size]
                U = self.squash(Z)  # [batch_size, K, 1, emb_size]
                delta_B = torch.matmul(low_capsule_iter, U.transpose(2, 3))  # [batch_size, K, seq_len, 1]
                delta_B = delta_B.reshape((-1, self.K, seq_len))
                B += delta_B  # [batch_size, K, seq_len]
            else:
                Z = torch.matmul(W, low_capsule_new)  # [batch_size, K, 1, emb_size]
                U = self.squash(Z)  # [batch_size, K, 1, emb_size]
        U = U.reshape((-1, self.K, self.emb_size))
        if self.relu_layer:
            U = self.layer(U)
        return U
