from torch.utils.data import Dataset
import pandas as pd

class AgeRecognitionDataset(Dataset):
    def __init__(self, triplet_csv_path, image_dir, preprocessor, cust_transforms=None):
        self.record = pd.read_csv(triplet_csv_path)
        self.image_dir = image_dir
        self.preprocessor = preprocessor
        self.cust_transforms = cust_transforms

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        entry = self.record.iloc[idx]
        anchor = self.preprocessor.preprocess(entry.anchor, cust_transforms=self.cust_transforms)
        positive = self.preprocessor.preprocess(entry.positive, cust_transforms=self.cust_transforms)
        negative = self.preprocessor.preprocess(entry.negative, cust_transforms=self.cust_transforms)
        return (anchor, positive, negative)

def kfold_cross_validation(dataset, curr_fold, folds=5):
    assert curr_fold < folds, "Current fold exceeds total number of folds"
    per_partition_length = len(dataset) // folds
    all_indices = np.arange(0, len(dataset))
    validating_indices = np.arange(curr_fold * per_partition_length, (curr_fold + 1) * per_partition_length)
    training_indices = np.setdiff1d(all_indices, validating_indices)
    return training_indices, validating_indices