

import os
import pickle


def cache_dataset(dataset, cache_path: str) -> None:
    """Caches a dataset to a specified path.

    Args:
        dataset (Dataset): The dataset to cache.
        cache_path (str): The path to cache the dataset to.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pickle.dump(dataset, open(cache_path, "wb"))
    
def load_dataset(cache_path: str):
    """Loads a dataset from a specified path.

    Args:
        cache_path (str): The path to load the dataset from.
    """
    return pickle.load(open(cache_path, "rb"))
