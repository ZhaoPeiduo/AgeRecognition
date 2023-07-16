from torch.utils.data import Dataset
import pandas as pd

class AgeRecognitionDataset(Dataset):
    def __init__(self, triplet_csv_path, image_dir, preprocesor, cust_transforms=None):
        self.record = pd.read_csv(triplet_csv_path)
        self.image_dir = image_dir
        self.preprocessor = preprocessor
        self.cust_transforms = cust_transforms

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        entry = self.record.iloc[idx]
        anchor = preprocessor.preprocess(entry.anchor, cust_transforms=self.cust_transforms)
        positive = preprocessor.preprocess(entry.positive, cust_transforms=self.cust_transforms)
        negative = preprocessor.preprocess(entry.negative, cust_transforms=self.cust_transforms)
        return (anchor, positive, negative)
