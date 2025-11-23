# using cc3m but selection only 10k image.
from datasets import load_dataset, Dataset

def get_dataset(limit: int = 0):
    ds_raw = load_dataset("conceptual_captions", "labeled", split="train")
    
    if limit > 0:
        ds_raw = ds_raw.select(range(limit)) # type: ignore

    ds = ds_raw.train_test_split(test_size=0.2, seed=42) # type: ignore

    train, test = ds["train"], ds["test"]
    return train, test