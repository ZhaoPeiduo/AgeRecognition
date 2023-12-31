import torch
import torch.nn as nn

def cosine_distancce(v1, v2):
    return 1 - nn.CosineSimilarity()(v1, v2) 

class AgeRecognitionLoss(nn.Module):
    def __init__(self, triplet_margin=1.0, cosine_embedding_margin=0.0, regularizing_strength_minimum=1e-5, device='cuda'):
        super(AgeRecognitionLoss, self).__init__()
        # Losses
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distancce, margin=triplet_margin)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=cosine_embedding_margin)
        # self.regularizing_strength = regularizing_strength
        self.regularizing_strength = nn.parameter.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.regularizing_strength_minimum = torch.tensor(regularizing_strength_minimum)
        self.device= device


    def forward(self, predictions):
        assert len(predictions.shape) in [2, 3], f"Predictions' shape does not match batch (3 dimensions) or invidual (2 dimension), found {predictions.shape}"
        if len(predictions.shape) == 3:
            anch_feat = predictions[:, 0, :]
            pos_feat = predictions[:, 1, :]
            neg_feat = predictions[:, 2, :]
            batch_size = predictions.shape[0]
        elif len(predictions.shape) == 2:
            anch_feat = predictions[0]
            pos_feat = predictions[1]
            neg_feat = predictions[2]
            batch_size = 1
        triplet_loss_value = self.triplet_loss(anch_feat, pos_feat, neg_feat)
        cosine_embedding_loss_value = self.cosine_embedding_loss(pos_feat, neg_feat, - torch.ones(batch_size).to(self.device))
        return  triplet_loss_value + torch.maximum(self.regularizing_strength, self.regularizing_strength_minimum) * cosine_embedding_loss_value
        