import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_data(df, config):
    train_df, test_df = train_test_split(
        df, test_size=config['test_size'],
        stratify=df["level"],
        random_state=config['seed']
    )
    val_ratio = config['val_size'] / (1 - config['test_size'])
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio,
        stratify=train_df["level"],
        random_state=config['seed']
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))
