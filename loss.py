import torch
import torch.nn as nn

class AgeRecognitionLoss(nn.Module):
    def __init__(self, triplet_margin=0.2, cosine_embedding_margin=0.2):
        super(AgeRecognitionLoss, self).__init__()
        # Losses
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=cosine_embedding_margin)
        # Learnable parameter
        self.importance = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)


    def forward(self, anch_feat, pos_feat, neg_feat):
        triplet_loss_value = self.triplet_loss(anch_feat, pos_feat, neg_feat)
        cosine_embedding_loss_value = self.cosine_embedding_loss(pos_feat, neg_feat, - torch.ones(pos_feat.shape[-1]))
        return  self.importance.item() * triplet_loss_value + (1 -  self.importance.item()) * cosine_embedding_loss_value
        