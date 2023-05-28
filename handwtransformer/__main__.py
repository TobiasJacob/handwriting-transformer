
import random
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path
from handwtransformer.config.Config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import animate_handwriting_sample, plot_handwriting_sample

if __name__ == "__main__":
    config = Config()
    dataset = load_iam_from_path(config.dataset_path, config.cache_path)
    print(len(dataset.samples))
    animate_handwriting_sample(dataset.samples[random.randint(0, len(dataset.samples))])
