class Config:
    dataset_path: str = "data/iam/"
    cache_path: str = "cache/iam.pkl"
    token_cache_path: str = "cache/iam_tokens.pkl"
    sequence_cache_path: str = "cache/iam_sequences.pkl"
    num_threads: int = 32
    device: str = "cuda:0"
