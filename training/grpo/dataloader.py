import pandas as pd
from torch.utils.data import Dataset
import json

class DescGenDataset(Dataset):
    def __init__(self, csv_path, split_ids):
        """
        Args:
            csv_path (str): Path to the CSV file.
            split_ids (list): List of IDs to include (train/val/test split).
        """
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['id'].isin(split_ids)]
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        features = row['features']

        # 解析 JSON
        if isinstance(features, str):
            features_list = json.loads(features)
            if isinstance(features_list, list):
                features = "; ".join(features_list)

        sample = {
            'id': row['id'],
            'title': row['title'],
            'features': features,
            'description': row['description']
        }
        return sample