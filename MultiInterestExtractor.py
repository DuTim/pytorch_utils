"""
# * @Author: DuTim
# * @Date: 2023-05-09 19:21:19
# * @LastEditTime: 2023-06-19 10:27:49
# * @Description: multi_head to extractor multi-interest
# * code from https://github.com/THUwangcy/ReChorus
"""

import torch
import torch.nn as nn
import numpy as np 
import utils.layer as layer
class MultiInterestExtractor(nn.Module):
    def __init__(self, k, emb_size,attn_size=5,max_his=20, add_pos=False, add_trm=False):
        super(MultiInterestExtractor, self).__init__()
        self.attn_size=attn_size
        self.max_his = max_his
        self.add_pos = add_pos
        self.add_trm = add_trm
        self.W1 = nn.Linear(emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, k)
        if self.add_trm:
            self.transformer = layer.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)
    def sequence_mask(self, lengths, max_len=None, dtype=torch.bool):
        """
        Pytorch equivalent for tf.sequence_mask.
        """
        if max_len is None:
            max_len = self.max_his
        row_vector = torch.arange(0, max_len, 1).to("cuda:0")
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask
    def forward(self, behavior_embs, pratical_seq_len,dropout_for_enhance=False):
        """
        Forward
            behavior_embs: [N, L, D]
        """

        batch_size, seq_len ,embedd_dim  = behavior_embs.shape

        valid_his = self.sequence_mask(pratical_seq_len,seq_len,dtype=torch.long)
      
        # his_vectors = self.i_embeddings(history) ## 256,20 ,64
        his_vectors = behavior_embs ## 256,20 ,64

        if self.add_trm:
            attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
            his_vectors = self.transformer(his_vectors, attn_mask)
            his_vectors = his_vectors * valid_his[:, :, None].float()

        # Multi-Interest Extraction
        attn_score = self.W2(self.W1(his_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb

        return interest_vectors

    
if __name__ == "__main__":
    torch.randn(128,20,10)
    MultiInterestExtractor()
