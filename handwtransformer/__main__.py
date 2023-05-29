
import random
from time import sleep

from matplotlib import pyplot as plt
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path
from handwtransformer.config.Config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import animate_handwriting_sample, plot_handwriting_sample
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor, sequence_tensor_to_handwriting_sample
from handwtransformer.offlinerl.text_converter import tokenize_dataset
from handwtransformer.offlinerl.train import train

if __name__ == "__main__":
    config = Config()

    # dataset = load_iam_from_path(config.dataset_path, config.cache_path)
    # print(len(dataset.samples))
    # # animate_handwriting_sample(dataset.samples[random.randint(0, len(dataset.samples))])
    
    # sequence_tensor, mask = generate_sequence_tensor(config)
    # text_tokens = tokenize_dataset(config)
    # print(sequence_tensor.shape)
    # print(mask.shape)
    # print(text_tokens[0].shape)
    
    # rand_i = random.randint(0, len(dataset.samples))
    # conversion_back = sequence_tensor_to_handwriting_sample(dataset.samples[rand_i].text, sequence_tensor[rand_i])
    # plot_handwriting_sample(conversion_back)
    # plt.show()
    # plot_handwriting_sample(dataset.samples[rand_i])
    # plt.show()
    
    train(config)
    
