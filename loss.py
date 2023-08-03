import torch
import torch.nn as nn

def cosine_distancce(v1, v2):
    return 1 - nn.CosineSimilarity()(v1, v2) 

class AgeRecognitionLoss(nn.Module):
    def __init__(self, triplet_margin=1.0, cosine_embedding_margin=0.0, device='cuda'):
        super(AgeRecognitionLoss, self).__init__()
        # Losses
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distancce, margin=triplet_margin)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=cosine_embedding_margin)
        # self.regularizing_strength = regularizing_strength
        self.regularizing_strength = nn.parameter.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.device= device


    def forward(self, predictions):
        assert len(predictions.shape) in [2, 3], f"Predictions' shape does not match batch (3 dimensions) or invidual (2 dimension), found {predictions.shape}"
        anch_feat = predictions[:, 0, :]
        pos_feat = predictions[:, 1, :]
        neg_feat = predictions[:, 2, :]
        batch_size = predictions.shape[0]
        triplet_loss_value = self.triplet_loss(anch_feat, pos_feat, neg_feat)
        cosine_embedding_loss_value = self.cosine_embedding_loss(pos_feat, neg_feat, - torch.ones(batch_size).to(self.device))
        return  triplet_loss_value + self.regularizing_strength * cosine_embedding_loss_value
        