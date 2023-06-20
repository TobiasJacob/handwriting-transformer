from typing import Optional, Tuple
import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, queryTokens, keyValueTokens, key_padding_mask=None, attn_mask=None):
        # Attention part
        x = queryTokens
        attn_out, _ = self.self_attn(queryTokens, keyValueTokens, keyValueTokens, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class HandwritingTransformer(torch.nn.Module):
    def __init__(self, max_seq_len: int, max_text_len: int, train_noise: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        encoding_size = 256
        self.max_seq_len = max_seq_len
        self.encoding_size = encoding_size
        self.n_mixtures = 20
        self.train_noise = train_noise
        
        self.letter_embedding = torch.nn.Embedding(256, encoding_size)
        # self.letter_positional_embedding = torch.nn.Parameter(torch.randn(max_text_len, encoding_size))
        self.letter_positional_embedding = torch.nn.Embedding(max_text_len, encoding_size)
        
        self.command_embedding = torch.nn.Linear(4, encoding_size)
        # self.command_positional_embedding = torch.nn.Parameter(torch.randn(self.max_seq_len, encoding_size))
        self.command_positional_embedding = torch.nn.Embedding(self.max_seq_len, encoding_size)
        
        self.block1 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        self.block2 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        self.block3 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        self.block4 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        self.block5 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        self.block6 = EncoderBlock(encoding_size, 8, 2 * encoding_size)
        # self.final_attention = torch.nn.MultiheadAttention(encoding_size, 8, batch_first=True)
        self.output = torch.nn.Linear(encoding_size, 2+5*self.n_mixtures)
        
    def forward(self, text: torch.Tensor, text_mask: torch.Tensor, handwriting: Optional[torch.Tensor], handwriting_mask: Optional[torch.Tensor], pred_mode=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            text (torch.Tensor): Text to be converted to handwriting. Shape == (batch_size, max_text_len)
            text_mask (torch.Tensor): Mask for the text. Shape == (batch_size, max_text_len)
            handwriting (Optional[torch.Tensor]): Ground truth sequences for training purposes. Shape == (batch_size, max_seq_length, 4)
            handwriting_mask (Optional[torch.Tensor]): Mask for the ground truth sequences. Shape == (batch_size, max_seq_length)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        handwriting_train = handwriting[:, 1:, :] # Shape == (batch_size, max_seq_len-1, 4)
        attn_mask = torch.triu(torch.ones(handwriting.shape[1], handwriting.shape[1], device=handwriting.device), diagonal=1) # Shape == (max_seq_len, max_seq_len)
        attn_mask = attn_mask.bool() # Shape == (max_seq_len, max_seq_len)
        text_mask = ~text_mask.bool() # Shape == (batch_size, max_text_len)
        handwriting_mask = ~handwriting_mask.bool() # Shape == (batch_size, max_seq_len)
        text = self.letter_embedding(text) # Shape == (batch_size, max_text_len, encoding_size)
        # Add positional encoding
        text = text + self.letter_positional_embedding(torch.arange(text.shape[1], device=text.device)) # Shape == (batch_size, max_text_len, encoding_size)
        # Apply self attention and feed forward block
        text = self.block1(text, text, key_padding_mask=text_mask) # Shape == (batch_size, max_text_len, encoding_size)
        text = self.block2(text, text, key_padding_mask=text_mask) # Shape == (batch_size, max_text_len, encoding_size)
        torch.cuda.synchronize()

        if self.training:
            handwriting[:, :, :2] = handwriting[:, :, :2] + torch.randn_like(handwriting[:, :, :2]) * self.train_noise
        torch.cuda.synchronize()
    
        handwriting = self.command_embedding(handwriting) # Shape == (batch_size, max_seq_len, encoding_size)
        command_encodings = self.command_positional_embedding(torch.arange(handwriting.shape[1], device=handwriting.device)) # Shape == (max_text_len, encoding_size)
        handwriting = handwriting + command_encodings # Shape == (batch_size, max_seq_len, encoding_size)
        torch.cuda.synchronize()
        
        # Command encodings
        handwriting = self.block3(handwriting, handwriting, key_padding_mask=handwriting_mask, attn_mask=attn_mask) # Shape == (batch_size, max_seq_length, encoding_size)
        handwriting = self.block4(handwriting, handwriting, key_padding_mask=handwriting_mask, attn_mask=attn_mask) # Shape == (batch_size, max_seq_length, encoding_size)
        z = self.block5(handwriting, text, key_padding_mask=text_mask) # Shape == (batch_size, max_seq_length, encoding_size)
        if pred_mode:
            z = self.block6(z[:, -1:], z, key_padding_mask=handwriting_mask) # Shape == (batch_size, 1, encoding_size
        else:
            z = self.block6(z, z, key_padding_mask=handwriting_mask, attn_mask=attn_mask) # Shape == (batch_size, max_seq_length, encoding_size)
        prob_params = self.output(z) # Shape == (batch_size, max_seq_length, 6)

        if not pred_mode:
            prob_params = prob_params[:, :-1, :] # Shape == (batch_size, max_seq_length-1, 6)
        torch.cuda.synchronize()
        
        # pen_delta is going to be a mixture model of 20 bivariate gaussians
        stroke_end = torch.distributions.Bernoulli(logits=prob_params[:, :, 0]) # Event_Shape == (batch_size, max_seq_length)
        episode_end = torch.distributions.Bernoulli(logits=prob_params[:, :, 1]) # Event_Shape == (batch_size, max_seq_length)
        
        n_mixtures = self.n_mixtures
        mixture_weights = prob_params[:, :, 2:2+n_mixtures] # Shape == (batch_size, max_seq_length, 20)
        gaussian_mus = prob_params[:, :, 2+n_mixtures:2+3*n_mixtures].reshape(prob_params.shape[0], prob_params.shape[1], n_mixtures, 2) # Shape == (batch_size, max_seq_length, 20, 2)
        gaussian_sigma = torch.exp(prob_params[:, :, 2+3*n_mixtures:2+5*n_mixtures].reshape(prob_params.shape[0], prob_params.shape[1], n_mixtures, 2)) # Shape == (batch_size, max_seq_length, 20, 2)
        
        mixture_distribution = torch.distributions.Categorical(logits=mixture_weights) # Event_Shape == (batch_size, max_seq_length, 20)
        component_distribution = torch.distributions.MultivariateNormal(gaussian_mus, torch.diag_embed(gaussian_sigma)) # Event_Shape == (batch_size, max_seq_length, 20, 2)
        
        pen_delta = torch.distributions.MixtureSameFamily(
            mixture_distribution,
            component_distribution
        ) # Event_Shape == (batch_size, max_seq_length, 2)
        torch.cuda.synchronize()
        
        # Compute predictions or loss
        if pred_mode:
            output = torch.zeros((handwriting.shape[0], 4), device=handwriting.device)
            output[:, :2] = pen_delta.sample()[:, -1, :] # Shape == (batch_size, 2)
            output[:, 2] = stroke_end.sample()[:, -1] # Shape == (batch_size,)
            output[:, 3] = episode_end.sample()[:, -1] # Shape == (batch_size,)
            torch.cuda.synchronize()
            return output
        else:
            torch.cuda.synchronize()
            loss = torch.zeros((handwriting_mask.shape[0], handwriting_mask.shape[1] - 1), device=handwriting_mask.device) # Shape == (batch_size, max_seq_len - 1)
            torch.cuda.synchronize()
            loss += -pen_delta.log_prob(handwriting_train[:, :, :2])
            torch.cuda.synchronize()
            loss += -stroke_end.log_prob(handwriting_train[:, :, 2])
            torch.cuda.synchronize()
            loss += -episode_end.log_prob(handwriting_train[:, :, 3])
            torch.cuda.synchronize()
            loss = (loss * ~handwriting_mask[:, 1:]).sum() / (~handwriting_mask[:, 1:]).sum()
            torch.cuda.synchronize()
            debug_stats = {
                "mixture_distribution_entropy": mixture_distribution.entropy().mean(),
                "component_distribution_entropy": component_distribution.entropy().mean(),
                "stroke_end_entropy": stroke_end.entropy().mean(),
                "episode_end_entropy": episode_end.entropy().mean(),
                "mse": ((pen_delta.mean - handwriting_train[:, :, :2])**2).mean(),
                "stroke_end_prob": stroke_end.probs.mean(),
                "episode_end_prob": episode_end.probs.mean(),
                "gaussian_sigma": gaussian_sigma.mean(),
                "gaussian_std": gaussian_sigma.mean().sqrt(),
                "gaussian_mus": gaussian_mus.mean(),
                "prob_params": prob_params.detach().cpu(),
                "episode_end": episode_end.probs.detach().cpu(),
                "stroke_end": stroke_end.probs.detach().cpu(),
            }
                
            torch.cuda.synchronize()
            return loss, debug_stats
