
import os
from typing import Tuple
import torch

from handwtransformer.dataset.Dataset import Dataset

def tokenize(text: str) -> torch.Tensor:
    """Tokenizes a text.

    Args:
        text (str): The text to tokenize.

    Returns:
        torch.Tensor: The tokenized text.
    """
    return torch.tensor([ord(c) for c in text])

def tokenize_dataset(dataset: Dataset, token_cache_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenizes a dataset.

    Args:
        dataset (Dataset): The dataset to tokenize.
        token_cache_path (str): The path to the cache file for the tokenized dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The tokenized dataset and the mask.
    """
    if os.path.exists(token_cache_path):
        return torch.load(token_cache_path)
    
    max_len = max([len(sample.text) for sample in dataset.samples])
    tokenized_dataset = torch.zeros((len(dataset.samples), max_len), dtype=torch.long)
    mask = torch.zeros((len(dataset.samples), max_len))
    for i, sample in enumerate(dataset.samples):
        tokens = tokenize(sample.text)
        tokenized_dataset[i, :len(tokens)] = tokens
        mask[i, :len(tokens)] = 1
    result = (tokenized_dataset, mask)
    torch.save(result, token_cache_path)
    return result

def detokenize(tokenized_text: torch.Tensor) -> str:
    """Detokenizes a tokenized text.

    Args:
        tokenized_text (torch.Tensor): The tokenized text.

    Returns:
        str: The detokenized text.
    """
    return "".join([chr(c) for c in tokenized_text])
    