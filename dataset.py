from torch.utils.data import Dataset, Subset
import torch
import pandas as pd
import numpy as np
import os

class AgeRecognitionDataset(Dataset):
    def __init__(self, triplet_csv_path, image_dir, preprocessor, cust_transforms=None, kfolds=5, device='cuda'):
        self.record = pd.read_csv(triplet_csv_path)
        self.image_dir = image_dir
        self.preprocessor = preprocessor
        self.cust_transforms = cust_transforms
        self.kfolds = kfolds
        self.device = device

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        entry = self.record.iloc[idx]
        anchor = self.preprocessor.preprocess(os.path.join(self.image_dir, entry.anchor))
        positive = self.preprocessor.preprocess(os.path.join(self.image_dir, entry.positive))
        negative = self.preprocessor.preprocess(os.path.join(self.image_dir, entry.negative))
        return torch.stack([anchor.to(self.device), positive.to(self.device), negative.to(self.device)])

    def kfold_cross_validation(self, curr_fold):
        assert curr_fold < self.kfolds, "Current fold exceeds total number of folds"
        per_partition_length = len(self) // self.kfolds
        all_indices = np.arange(0, len(self))
        validating_indices = np.arange(curr_fold * per_partition_length, (curr_fold + 1) * per_partition_length)
        training_indices = np.setdiff1d(all_indices, validating_indices)
        validation_set = Subset(self, validating_indices)
        training_set = Subset(self, training_indices)
        return training_set, validation_set