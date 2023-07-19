import torch

from preprocessor import AgeRecognitionPreprocessor
from dataset import AgeRecognitionDataset
from models import vit_l_16_age_recognizer
from loss import AgeRecognitionLoss



def train():
    model = vit_l_16_age_recognizer()
    optimizer = torch.optim.Adam(model.params)