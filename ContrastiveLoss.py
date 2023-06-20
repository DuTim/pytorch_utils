"""
# * @Author: DuTim
# * @Date: 2023-06-20 19:21:38
# * @LastEditTime: 2023-06-20 19:29:18
# * @Description: Contrastive loss
"""
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    More than inspired from https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py

    Notes
    =====

    Using this pytorch implementation, you don't actually need to l2-norm the inputs, the results will be
    identical, as shown if you run this file.
    """

    def __init__(self, batch_size, temperature, device):

        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.get_correlated_samples_mask()
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss

    def get_correlated_samples_mask(self):
        mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask


# if __name__ == "__main__":
#     a, b = torch.rand(8, 12), torch.rand(8, 12)
#     a_norm, b_norm = torch.nn.functional.normalize(a), torch.nn.functional.normalize(b)
#     cosine_sim = torch.nn.CosineSimilarity()
#     ntxent_loss = NT_Xent(8, 0.5, "cpu")
#     loss1=torch.allclose(cosine_sim(a, b), cosine_sim(a_norm, b_norm))
#     loss2=torch.allclose(ntxent_loss(a, b), ntxent_loss(a_norm, b_norm))
# ntxent_loss(a, b), ntxent_loss(a_norm, b_norm)
