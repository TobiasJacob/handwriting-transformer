import torch
from handwtransformer.config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import plot_handwriting_sample
from handwtransformer.offlinerl.model import HandwritingTransformer
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor, sequence_tensor_to_handwriting_sample
from handwtransformer.offlinerl.text_converter import detokenize, tokenize_dataset

from torch.utils.tensorboard import SummaryWriter

def train(config: Config):
    sequences, sequences_mask = generate_sequence_tensor(config)
    tokens, tokens_mask = tokenize_dataset(config)
    
    model = HandwritingTransformer(max_seq_len=sequences.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 32
    pred_window = 32
    preview_batch_size = 8
    
    summary_writer = SummaryWriter()
    
    for step in range(1000):
        for mode in ["train", "train", "train", "train", "eval"]:
            split_i = len(sequences) // 10 * 8
            if mode == "train":
                batch_indices = torch.randint(0, split_i, (batch_size,))
            else:
                batch_indices = torch.randint(split_i, len(sequences), (batch_size,))
            predict_cutoff = torch.randint(0, sequences.shape[1] - pred_window, (1,))
            
            batch_tokens = tokens[batch_indices]
            batch_tokens_mask = tokens_mask[batch_indices]
            batch_sequences_so_far = sequences[batch_indices, :predict_cutoff]
            batch_sequences_so_far_mask = sequences_mask[batch_indices, :predict_cutoff]
            train_sequences = sequences[batch_indices, predict_cutoff:predict_cutoff + pred_window]
            train_sequences_mask = sequences_mask[batch_indices, predict_cutoff:predict_cutoff + pred_window]
            
            if train_sequences_mask.sum() > 0:
                if mode == "eval":
                    with torch.no_grad():
                        model.eval()
                        loss = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, train_sequences, train_sequences_mask)
                else:
                    model.train()
                    loss = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, train_sequences, train_sequences_mask)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                print(step, loss.item())
                summary_writer.add_scalar(f"{mode}/loss", loss.item(), step)
                    
            if step % 10 == 9 and mode == "eval":
                # generate some samples
                model.eval()
                eval_indices = batch_indices[:preview_batch_size]
                batch_tokens = tokens[eval_indices]
                batch_tokens_mask = tokens_mask[eval_indices]
                batch_sequences_so_far = torch.zeros_like(sequences[eval_indices])
                batch_sequences_so_far_mask = torch.ones_like(sequences_mask[eval_indices])
                for i in range(0, sequences.shape[1] - pred_window + 1, pred_window):
                    print("Generating preview, time step ", i)
                    # temp fix to reduce runtime
                    if i > 300:
                        break
                    train_sequences = batch_sequences_so_far[:, i:i + pred_window]
                    train_sequences_mask = batch_sequences_so_far_mask[:, i:i + pred_window]
                    pred = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, None, None)
                    batch_sequences_so_far[:, i:i + pred_window] = pred
                    # update the mask starting from the second prediction
                    for t in range(0 if i > 0 else 1, pred.shape[1]):
                        # use elementwise and to set the mask to 1 only if it was 1 before and the 3rd element of the prediction is not 1
                        batch_sequences_so_far_mask[:, i + t] = batch_sequences_so_far_mask[:, i + t] & (batch_sequences_so_far[:, i + t - 1, 2] > 0.5)
                for i in range(preview_batch_size):
                    fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), batch_sequences_so_far[i]))
                    # Save fig to tensorboard
                    summary_writer.add_figure(f"sample_{i}", fig, step)
        
        