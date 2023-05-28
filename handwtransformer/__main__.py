
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path
from handwtransformer.config.Config import Config

if __name__ == "__main__":
    config = Config()
    dataset = load_iam_from_path(config.dataset_path, config.cache_path)
    print(len(dataset.samples))
