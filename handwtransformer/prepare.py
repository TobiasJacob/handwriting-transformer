
import random
from time import sleep

from matplotlib import pyplot as plt
import torch
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path
from handwtransformer.config.Config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import animate_handwriting_sample, plot_handwriting_sample
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor, sequence_tensor_to_handwriting_sample
from handwtransformer.offlinerl.text_converter import tokenize_dataset
from handwtransformer.offlinerl.train import train

if __name__ == "__main__":
    config = Config()
    print("Loading dataset...")

    dataset = load_iam_from_path(config.dataset_path, config.cache_path)
    
    sequence_tensor, mask = generate_sequence_tensor(config)
    text_tokens = tokenize_dataset(config)

    print("Done")
