import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from src.config import DATA_DIR
from pathlib import Path

def load_metadata():
    path = DATA_DIR / "meta_data.csv"
    df   = pd.read_csv(path)

    # Store only filename — full path constructed at runtime in BRARDataset   
    df["image_path"] = df['image_path'].apply(
    lambda x: Path(str(x).replace('\\', '/')).name
    )
    
    return df

class BRARDataset(Dataset):               
    def __init__(self, df, config, transform=None, norm_stats=None):
        self.df        = df.reset_index(drop=True).copy()
        self.transform = transform
    
        # Use config metadata_cols — supports sensitivity analysis
        self.meta_cols      = config.get('metadata_cols', [
            "age", "gender", "number_of_missing_teeth",
            "implant", "residual_root", "functional_tooth_logarithm"
        ])
        self.normalize_cols = config.get('normalize_cols', [
            "age", "number_of_missing_teeth", "implant",
            "residual_root", "functional_tooth_logarithm"
        ])
        # Only normalize cols that are in meta_cols
        self.normalize_cols = [c for c in self.normalize_cols
                                if c in self.meta_cols]
    
        if norm_stats is None:
            self.norm_stats = {col: (self.df[col].min(), self.df[col].max())
                               for col in self.normalize_cols}
        else:
            self.norm_stats = norm_stats
    
        for col in self.normalize_cols:
            col_min, col_max = self.norm_stats[col]
            if col_max > col_min:
                self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
            
    def get_norm_stats(self):
        return self.norm_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path from DATA_DIR + level + filename
        filename = self.df.loc[idx, "image_path"]
        level    = int(self.df.loc[idx, "level"])
        image_path = DATA_DIR / "manual_cropped_dataset" / \
                     f"level_{level}" / filename
        
        image = Image.open(image_path).convert("RGB")
        label = level - 1
        if self.transform:
            image = self.transform(image)
        meta = torch.tensor(
            self.df.loc[idx, self.meta_cols].values.astype(float),
            dtype=torch.float32
        )
        return image, meta, label
    
class BRARDatasetMeta(Dataset):
    def __init__(self, df, config, transform=None, norm_stats=None):
        self.df = df.reset_index(drop=True).copy()
        h, w = config['input_size']
        self.zero_image = torch.zeros(3, h, w)

        # Use config metadata_cols — supports sensitivity analysis
        self.meta_cols      = config.get('metadata_cols', [
            "age", "gender", "number_of_missing_teeth",
            "implant", "residual_root", "functional_tooth_logarithm"
        ])
        self.normalize_cols = config.get('normalize_cols', [
            "age", "number_of_missing_teeth", "implant",
            "residual_root", "functional_tooth_logarithm"
        ])
        # Only normalize cols that are in meta_cols
        self.normalize_cols = [c for c in self.normalize_cols
                                if c in self.meta_cols]

        if norm_stats is None:
            self.norm_stats = {col: (self.df[col].min(), self.df[col].max())
                               for col in self.normalize_cols}
        else:
            self.norm_stats = norm_stats

        for col in self.normalize_cols:
            col_min, col_max = self.norm_stats[col]
            if col_max > col_min:
                self.df[col] = (self.df[col] - col_min) / (col_max - col_min)

    def get_norm_stats(self):
        return self.norm_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.loc[idx, "level"]) - 1
        meta  = torch.tensor(
            self.df.loc[idx, self.meta_cols].values.astype(float),
            dtype=torch.float32
        )
        return self.zero_image, meta, label

def get_transforms(config, train=True):
    version = config.get('transform_version', 'default')
    
    base = [
        transforms.Resize(config['input_size']),
    ]

    if train:
        if version == 'v1':
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAutocontrast(p=0.2),
            ]
        elif version == 'v2':
            aug = []
        else:
            aug = [
                transforms.RandomRotation(2),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            ]

        return transforms.Compose(
            base + aug + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        base + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

def build_datasets(train_df, val_df, test_df, config):
    """
    Factory — returns (train_dataset, val_dataset, test_dataset, norm_stats)
    based on fusion_mode in config.
    meta_only  → BRARDatasetMeta (no image loading)
    image_only → BRARDataset
    multimodal → BRARDataset
    """
    if config['fusion_mode'] == 'meta_only':
        train_dataset = BRARDatasetMeta(train_df, config)
        norm_stats    = train_dataset.get_norm_stats()
        val_dataset   = BRARDatasetMeta(val_df,   config, norm_stats=norm_stats)
        test_dataset  = BRARDatasetMeta(test_df,  config, norm_stats=norm_stats)
    else:
        train_dataset = BRARDataset(train_df, config,
                                    transform=get_transforms(config, train=True))
        norm_stats    = train_dataset.get_norm_stats()
        val_dataset   = BRARDataset(val_df,   config,
                                    transform=get_transforms(config, train=False),
                                    norm_stats=norm_stats)
        test_dataset  = BRARDataset(test_df,  config,
                                    transform=get_transforms(config, train=False),
                                    norm_stats=norm_stats)

    return train_dataset, val_dataset, test_dataset, norm_stats

def get_dataloaders(train_dataset, val_dataset, test_dataset, config):
    num_workers = config.get('num_workers', 2)  # default 2 if not in config
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader


