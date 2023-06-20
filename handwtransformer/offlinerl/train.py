from matplotlib import pyplot as plt
import numpy as np
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
    handwriting, handwriting_mask = generate_sequence_tensor(config)
    # prepend start token as 0 to the handwriting
    handwriting = torch.cat([torch.zeros((handwriting.shape[0], 1, handwriting.shape[2]), dtype=handwriting.dtype), handwriting], dim=1)
    handwriting_mask = torch.cat([torch.ones((handwriting_mask.shape[0], 1), dtype=handwriting_mask.dtype), handwriting_mask], dim=1)
    tokens, tokens_mask = tokenize_dataset(config)
    handwriting = handwriting.to(config.device)
    handwriting_mask = handwriting_mask.to(config.device)
    print("Std:", torch.std(handwriting[handwriting_mask][:2]))
    tokens = tokens.to(config.device)
    tokens_mask = tokens_mask.to(config.device)
    
    train_noise = 2.0
    model = HandwritingTransformer(max_seq_len=handwriting.shape[1], max_text_len = tokens.shape[1], train_noise = train_noise)
    model.to(config.device)
    print(model)
    print("Num parameters:", sum([p.numel() for p in model.parameters()]))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 16
    preview_batch_size = 8
    
    summary_writer = SummaryWriter()
    
    for step in range(100000):
        for mode in ["train", "train", "train", "train", "eval"]:
            train_handwriting_mask = None
            split_i = len(handwriting) // 10 * 8
            
            if mode == "train": 
                # choose until split_i but make no duplicates
                batch_indices_np = np.random.choice(split_i, batch_size, replace=False)
                batch_indices = torch.from_numpy(batch_indices_np).to(tokens.device)
            else:
                batch_indices_np = np.random.choice(len(handwriting) - split_i, batch_size, replace=False) + split_i
                batch_indices = torch.from_numpy(batch_indices_np).to(tokens.device)
    
            while train_handwriting_mask is None or train_handwriting_mask.sum() == 0:
                batch_tokens = tokens[batch_indices]
                batch_tokens_mask = tokens_mask[batch_indices]
                train_handwriting = handwriting[batch_indices]
                train_handwriting_mask = handwriting_mask[batch_indices]
            
            if mode == "eval":
                with torch.no_grad():
                    model.eval()
                    loss, debug_stats = model(batch_tokens, batch_tokens_mask, train_handwriting, train_handwriting_mask)
            else:
                model.train()
                loss, debug_stats = model(batch_tokens, batch_tokens_mask, train_handwriting, train_handwriting_mask)
                optim.zero_grad()
                loss.backward()
                optim.step()
            for name, value in debug_stats.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    summary_writer.add_scalar(f"{mode}/{name}", value, step)
            # print(step, mode, loss.item())
            summary_writer.add_scalar(f"{mode}/loss", loss.cpu().item(), step)
                
        if step % 30 == 0:
            for mode in ["train", "eval"]:
                # generate some samples
                if mode == "train":
                    model.train()
                else:
                    model.eval()
                eval_indices = torch.randint(split_i, len(handwriting), (preview_batch_size,), device=tokens.device)
                batch_tokens = tokens[eval_indices]
                batch_tokens_mask = tokens_mask[eval_indices]
                gen_handwriting = torch.zeros_like(handwriting[eval_indices])
                gen_handwriting_mask = torch.zeros_like(handwriting_mask[eval_indices], dtype=torch.bool)
                gen_handwriting_mask[:, 0] = True
                for i in range(1, handwriting.shape[1] - 1):
                    # if i % 20 == 0:
                    #     print("Generating preview, time step ", i)
                    # temp fix to reduce runtime
                    if i > 300:
                        break
                    pred = model(batch_tokens, batch_tokens_mask, gen_handwriting[:, :i], gen_handwriting_mask[:, :i], pred_mode=True)
                    gen_handwriting[:, i] = pred
                    # use elementwise and to set the mask to 1 only if it was 1 before and the 4rd element of the prediction is not 1
                    gen_handwriting_mask[:, i] = gen_handwriting_mask[:, i - 1] & (gen_handwriting[:, i - 1, 3] < 0.5)
                for i in range(preview_batch_size):
                    # plot the generated sample
                    fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), gen_handwriting[i]))
                    summary_writer.add_figure(f"sample_{mode}_{i}", fig, step)
                # plot the training data
                fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), handwriting[eval_indices[i]]))
                summary_writer.add_figure(f"sample_{mode}_groundtruth", fig, step)
                sample_with_noise = handwriting[eval_indices[i]].clone()
                sample_with_noise[:, :2] += torch.randn_like(sample_with_noise[:, :2]) * train_noise
                torch.cuda.synchronize()
                fig = plot_handwriting_sample(sequence_tensor_to_handwriting_sample(detokenize(batch_tokens[i]), sample_with_noise))
                summary_writer.add_figure(f"sample_{mode}_groundtruth_with_noise", fig, step)
                # also log the debug stats
                # imshow for the prop params of the first sample
                summary_writer.add_image(f"{mode}/prob_params", debug_stats["prob_params"][0], step, dataformats="HW")
                # add the probs for episode end as line plot episode_end and stroke_end
                fig = plt.figure()
                plt.plot(torch.mean(debug_stats["episode_end"], dim=0).cpu().numpy())
                summary_writer.add_figure(f"{mode}/episode_end", fig, step)
                fig = plt.figure()
                plt.plot(torch.mean(debug_stats["stroke_end"], dim=0).cpu().numpy())
                summary_writer.add_figure(f"{mode}/stroke_end", fig, step)
                fig = plt.figure()
                plt.plot(handwriting[:, :, 3].sum(dim=0).cpu().numpy())
                summary_writer.add_figure(f"{mode}/stroke_end_groundtruth", fig, step)
                # log an example tensor of what the model is predicting as text
                summary_writer.add_text(f"{mode}/tensor_prediction", str(gen_handwriting[0, 0:200]), step)
                # log an example tensor of what the model should predict as text
                summary_writer.add_text(f"{mode}/tensor_groundtruth", str(handwriting[eval_indices][0, 0:200]), step)
        
        if step % 100 == 0:
            # save the model in summary writer path
            path = summary_writer.log_dir + f"/model_{step}.pth"
            torch.save(model.state_dict(), path)
