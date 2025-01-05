import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRankingLoss(nn.Module):
    """
        This loss works as the following:
            1. For labels 1 and -1, it computes the standart margin ranking loss:
                loss = max(0, margin - (f_1 - f_2)) - means we penalize if the difference is lower than 1.
            2. For labels 0.5 and -0.5, we penalize only if the sign is wrong:
            3. If the label is 0, we penalize if the difference is bigger than 1

            2. If the label is 0, it means that we want this pair to be close to each other, and we penalize it if the difference is bigger than threshold.
    """

    def __init__(self, loss_weights=[1, 1]):
        """
            Args:
                padding - padding from the 0.5 when we start to penalize incorrect scoring
                loss_weights - weights for the loss of the different types of labels
        """

        super(CustomRankingLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, f_first, f_second, labels):
        """
            f_first, f_second - tensors of scores of the first and second texts in pairs
            labels - margins for the first and second texts that obtained through LLM model

        """
        diff = f_first - f_second
        loss = torch.zeros_like(labels)
        diff = diff * labels.sign()

        mask_05 = (labels == 0.5) | (labels == -0.5)
        loss_05 = torch.relu(0 - diff) + torch.relu(diff - 1)

        mask_1 = (labels == 1) | (labels == -1)
        loss_1 = torch.relu(1 - diff)

        mask_0 = (labels == 0)
        loss_0 = torch.relu(-1 - diff) + torch.relu(diff - 1)

        loss = self.loss_weights[0] * loss_05[mask_05].sum() + self.loss_weights[1] * loss_1[mask_1].sum() + loss_0[mask_0].sum()

        return loss