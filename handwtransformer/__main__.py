
import random
from handwtransformer.dataset.loaders.IamLoader import load_iam_from_path
from handwtransformer.config.Config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import animate_handwriting_sample, plot_handwriting_sample
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor, sequence_tensor_to_handwriting_sample
from handwtransformer.offlinerl.text_converter import tokenize_dataset

if __name__ == "__main__":
    config = Config()
    dataset = load_iam_from_path(config.dataset_path, config.cache_path)
    print(len(dataset.samples))
    # animate_handwriting_sample(dataset.samples[random.randint(0, len(dataset.samples))])
    
    sequence_tensor, mask = generate_sequence_tensor(dataset, config.sequence_cache_path)
    text_tokens = tokenize_dataset(dataset, config.token_cache_path)
    rand_i = random.randint(0, len(dataset.samples))
    conversion_back = sequence_tensor_to_handwriting_sample(dataset.samples[rand_i].text, sequence_tensor[rand_i])
    print(sequence_tensor[rand_i])
    print(mask[rand_i])
    print(conversion_back)
    plot_handwriting_sample(conversion_back)
    plot_handwriting_sample(dataset.samples[rand_i])
    
    
