import os
from typing import Tuple
import torch
from handwtransformer.dataset.Dataset import Dataset
from handwtransformer.dataset.HandwritingSample import HandwritingSample
from handwtransformer.config.Config import Config
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path


def generate_sequence_tensor(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a sequence tensor from a dataset.

    Args:
        dataset (Dataset): The dataset to generate the sequence tensor from.
        config (Config): The config.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The sequence tensor(num_samples, max_seq_len, 4) and a mask(num_samples, max_seq_len).
    """
    
    if os.path.exists(config.sequence_cache_path):
        return torch.load(config.sequence_cache_path)
    
    dataset = load_iam_from_path(config.dataset_path, config.cache_path)

    max_seq_len = max([sum([stroke.shape[0] for stroke in sample.strokes]) for sample in dataset.samples])
    sequence_tensor = torch.zeros((len(dataset.samples), max_seq_len, 4))
    mask = torch.zeros((len(dataset.samples), max_seq_len))
    for i, sample in enumerate(dataset.samples):
        j = 0
        for stroke in sample.strokes:
            stroke_len = stroke.shape[0]
            
            sequence_tensor[i, j:j + stroke_len, :2] = torch.from_numpy(stroke)
            mask[i, j:j + stroke_len] = 1
            sequence_tensor[i, j + stroke_len - 1, 2] = 1
            
            j += stroke.shape[0]
        sequence_tensor[i, j - 1, 3] = 1

    # Calcualte deltas and normalize
    sequence_tensor[:, :-1, :2] = sequence_tensor[:, 1:, :2] - sequence_tensor[:, :-1, :2]
    sequence_tensor[:, :, :2] = sequence_tensor[:, :, :2] / torch.std(sequence_tensor[:, :, :2], dim=(0, 1), keepdim=True)
    sequence_tensor[:, -1, :2] = 0
    
    result = (sequence_tensor, mask)
    torch.save(result, config.sequence_cache_path)
    return result

def sequence_tensor_to_handwriting_sample(text: str, sequence_tensor: torch.Tensor) -> HandwritingSample:
    """Converts a sequence tensor to a handwriting sample.

    Args:
        text (str): The text of the handwriting sample.
        sequence_tensor (torch.Tensor): The sequence tensor to convert. Shape == (max_seq_len, 4)

    Returns:
        HandwritingSample: The converted handwriting sample.
    """
        
    assert sequence_tensor.shape[1] == 4
    assert len(sequence_tensor.shape) == 2
    abs_poses = sequence_tensor[:, :2].cumsum(dim=0)
    strokes = []
    last_i = 0
    for i in range(sequence_tensor.shape[0]):
        if sequence_tensor[i, 2] == 1:
            strokes.append(abs_poses[last_i:i].cpu().numpy())
            last_i = i
        if sequence_tensor[i, 3] == 1:
            break
    return HandwritingSample(text, strokes)
