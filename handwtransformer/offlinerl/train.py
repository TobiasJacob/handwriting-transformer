import torch
from handwtransformer.config import Config
from handwtransformer.dataset.plotters.HandwritingSamplePlotter import plot_handwriting_sample
from handwtransformer.offlinerl.model import HandwritingTransformer
from handwtransformer.offlinerl.sequence_converter import generate_sequence_tensor, sequence_tensor_to_handwriting_sample
from handwtransformer.offlinerl.text_converter import detokenize, tokenize_dataset

from torch.utils.tensorboard import SummaryWriter

def train(config: Config):
    print("Num threads:", torch.get_num_threads())
    print("Num GPUs:", torch.cuda.device_count())
    sequences, sequences_mask = generate_sequence_tensor(config)
    tokens, tokens_mask = tokenize_dataset(config)
    sequences = sequences.to(config.device)
    sequences_mask = sequences_mask.to(config.device)
    print("Std:", torch.std(sequences[sequences_mask][:2]))
    tokens = tokens.to(config.device)
    tokens_mask = tokens_mask.to(config.device)
    
    model = HandwritingTransformer(max_seq_len=sequences.shape[1])
    model.to(config.device)
    print(model)
    print("Num parameters:", sum([p.numel() for p in model.parameters()]))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    batch_size = 16
    preview_batch_size = 8
    
    summary_writer = SummaryWriter()
    
    for step in range(1000):
        for mode in ["train", "train", "train", "train", "eval"]:
            train_sequences_mask = None
            split_i = len(sequences) // 10 * 8
            
            if mode == "train":
                batch_indices = torch.randint(0, split_i, (batch_size,), device=tokens.device)
            else:
                batch_indices = torch.randint(split_i, len(sequences), (batch_size,), device=tokens.device)
    
            while train_sequences_mask is None or train_sequences_mask.sum() == 0:
                predict_cutoff = torch.randint(0, sequences.shape[1] - 1, (1,), device=tokens.device)
                batch_tokens = tokens[batch_indices]
                batch_tokens_mask = tokens_mask[batch_indices]
                batch_sequences_so_far = sequences[batch_indices, :predict_cutoff]
                batch_sequences_so_far_mask = sequences_mask[batch_indices, :predict_cutoff]
                train_sequences = sequences[batch_indices, predict_cutoff:predict_cutoff + 1]
                train_sequences_mask = sequences_mask[batch_indices, predict_cutoff:predict_cutoff + 1]
            
            if mode == "eval":
                with torch.no_grad():
                    model.eval()
                    loss, debug_stats = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, train_sequences, train_sequences_mask)
            else:
                model.train()
                loss, debug_stats = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far, batch_sequences_so_far_mask, train_sequences, train_sequences_mask)
                optim.zero_grad()
                loss.backward()
                optim.step()
            # print(step, mode, loss.item())
            summary_writer.add_scalar(f"{mode}/loss", loss.cpu().item(), step)
                
        if step % 10 == 9:
            for mode in ["train", "eval"]:
                # generate some samples
                if mode == "train":
                    model.train()
                else:
                    model.eval()
                eval_indices = torch.randint(split_i, len(sequences), (preview_batch_size,), device=tokens.device)
                batch_tokens = tokens[eval_indices]
                batch_tokens_mask = tokens_mask[eval_indices]
                batch_sequences_so_far = torch.zeros_like(sequences[eval_indices])
                batch_sequences_so_far_mask = torch.ones_like(sequences_mask[eval_indices])
                for i in range(0, sequences.shape[1] - 1):
                    # if i % 20 == 0:
                    #     print("Generating preview, time step ", i)
                    # temp fix to reduce runtime
                    if i > 300:
                        break
                    pred = model(batch_tokens, batch_tokens_mask, batch_sequences_so_far[:, :i], batch_sequences_so_far_mask[:, :i], None, None)
                    batch_sequences_so_far[:, i:i + 1] = pred
                    # update the mask starting from the second prediction
                    for t in range(0 if i > 0 else 1, pred.shape[1]):
                        # use elementwise and to set the mask to 1 only if it was 1 before and the 3rd element of the prediction is not 1
                        batch_sequences_so_far_mask[:, i + t] = batch_sequences_so_far_mask[:, i + t - 1] & (batch_sequences_so_far[:, i + t - 1, 2] > 0.5)
                for i in range(preview_batch_size):
                    # plot the generated sample
                    fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), batch_sequences_so_far[i]))
                    summary_writer.add_figure(f"sample_{mode}_{i}", fig, step)
                # plot the training data
                fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), sequences[eval_indices[i]]))
                summary_writer.add_figure(f"sample_{mode}_groundtruth", fig, step)
                # also log the debug stats
                for name, value in debug_stats.items():
                    summary_writer.add_scalar(f"{mode}/{name}", value, step)
                # log an example tensor of what the model is predicting as text
                summary_writer.add_text(f"{mode}/tensor_prediction", str(batch_sequences_so_far[0, 0:200]), step)
                # log an example tensor of what the model should predict as text
                summary_writer.add_text(f"{mode}/tensor_groundtruth", str(sequences[eval_indices][0, 0:200]), step)
        