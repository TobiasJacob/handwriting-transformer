import torch
from handwtransformer.config import Config
from handwtransformer.offlinerl.model import HandwritingTransformer
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor
from handwtransformer.offlinerl.text_converter import tokenize_dataset

# from torch.utils.tensorboard import SummaryWriter

def train(config: Config):
    sequences, sequences_mask = generate_sequence_tensor(config)
    tokens, tokens_mask = tokenize_dataset(config)
    
    model = HandwritingTransformer(max_seq_len=sequences.shape[1])
    batch_size = 32
    pred_window = 32
    
    # summary_writer = SummaryWriter()
    
    for step in range(1000):
        batch_indices = torch.randint(0, len(sequences), (batch_size,))
        predict_cutoff = torch.randint(0, sequences.shape[1] - pred_window, (1,))
        
        batch_tokens = tokens[batch_indices]
        batch_tokens_mask = tokens_mask[batch_indices]
        batch_sequences_so_far = sequences[batch_indices, :predict_cutoff]
        batch_sequences_so_far_mask = sequences_mask[batch_indices, :predict_cutoff]
        train_sequences = sequences[batch_indices, predict_cutoff:predict_cutoff + pred_window]
        train_sequences_mask = sequences_mask[batch_indices, predict_cutoff:predict_cutoff + pred_window]
        
        if train_sequences_mask.sum() == 0:
            continue
                
        loss = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, train_sequences, train_sequences_mask)
        
        print(step, loss.item())
        # summary_writer.add_scalar("loss", loss, step)
    