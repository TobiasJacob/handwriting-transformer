from typing import Optional
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

    def forward(self, x, mask=None):
        # Attention part
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class HandwritingTransformer(torch.nn.Module):
    def __init__(self, max_seq_len: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        encoding_size = 64
        self.max_seq_len = max_seq_len
        self.encoding_size = encoding_size
        
        self.letter_embedding = torch.nn.Embedding(256, encoding_size)
        self.letter_positional_embedding = torch.nn.Embedding(self.max_seq_len, encoding_size)
        
        self.command_embedding = torch.nn.Linear(4, encoding_size)
        self.command_positional_embedding = torch.nn.Embedding(self.max_seq_len, encoding_size)
        
        self.block1 = EncoderBlock(encoding_size, 8, 128)
        self.block2 = EncoderBlock(encoding_size, 8, 128)
        self.block3 = EncoderBlock(encoding_size, 8, 128)
        self.final_attention = torch.nn.MultiheadAttention(encoding_size, 8, batch_first=True)
        self.output = torch.nn.Linear(encoding_size, 6)
        
    def forward(self, text: torch.Tensor, text_mask: torch.Tensor, sequences: Optional[torch.Tensor], sequences_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the handwriting transformer.

        Args:
            text (torch.Tensor): The text to generate the handwriting for. Shape == (batch_size, max_text_len)

        Returns:
            torch.Tensor: The generated handwriting. Shape == (batch_size, max_seq_len, 4)
        """
        x = self.letter_embedding(text) # Shape == (batch_size, max_text_len, encoding_size)
        # Add positional encoding
        x = x + self.letter_positional_embedding(torch.arange(x.shape[1], device=x.device)) # Shape == (batch_size, max_text_len, encoding_size)
        # Apply self attention and feed forward block
        x = self.block1(x, text_mask) # Shape == (batch_size, max_text_len, encoding_size)
        
        command_encodings = self.command_positional_embedding(torch.arange(self.max_seq_len, device=x.device)) # Shape == (max_text_len, encoding_size)
        y = torch.zeros((x.shape[0], self.max_seq_len, 4), device=x.device)
        latent = torch.zeros((x.shape[0], x.shape[1] + self.max_seq_len, self.encoding_size), device=x.device) # Shape == (batch_size, max_text_len + max_seq_len, encoding_size)
        
        latent[:, :x.shape[1]] = x # Shape == (batch_size, max_text_len + max_seq_len, encoding_size)
        if sequences is not None:
            loss = torch.zeros((x.shape[0], self.max_seq_len), device=x.device)
        for i in range(self.max_seq_len):
            print(i)
            # Command encodings
            if i > 0:
                latent[:, x.shape[1] + i] += self.command_embedding(y[:, i - 1]) # Shape == (batch_size, encoding_size)
            latent[:, x.shape[1] + i] += command_encodings[i] # Shape == (batch_size, encoding_size)
            
            
            z = self.block2(latent[:, :x.shape[1] + i]) # Shape == (batch_size, max_text_len + i, encoding_size)
            z = self.block3(z) # Shape == (batch_size, max_text_len + i, encoding_size)
            z, _ = self.final_attention(torch.zeros_like(z[:, 0:1]), z, z) # Shape == (batch_size, max_text_len + i, encoding_size)
            prob_params = self.output(z) # Shape == (batch_size, 6)
            
            pen_delta = torch.distributions.MultivariateNormal(prob_params[:, 0, :2], torch.diag_embed(torch.exp(prob_params[:, 0, 2:4]))) # Event_Shape == (batch_size, 2)
            stroke_end = torch.distributions.Bernoulli(torch.sigmoid(prob_params[:, 0, 4])) # Event_Shape == (batch_size,)
            episode_end = torch.distributions.Bernoulli(torch.sigmoid(prob_params[:, 0, 5])) # Event_Shape == (batch_size,)
            
            if sequences is None:
                y[:, i, :2] = pen_delta.sample() # Shape == (batch_size, max_seq_len, 2)
                y[:, i, 2] = stroke_end.sample() # Shape == (batch_size, max_seq_len)
                y[:, i, 3] = episode_end.sample() # Shape == (batch_size, max_seq_len)
            else:
                y[:, i, :2] = sequences[:, i - 1, :2] + pen_delta.sample() # TODO: Check if i - 1 helps, or if it should be i and no call to sample()
                y[:, i, 2] = sequences[:, i - 1, 2] + stroke_end.sample()
                y[:, i, 3] = sequences[:, i - 1, 3] + episode_end.sample()
                loss[:, i] += pen_delta.log_prob(sequences[:, i - 1, :2])
                loss[:, i] += stroke_end.log_prob(sequences[:, i - 1, 2])
                loss[:, i] += episode_end.log_prob(sequences[:, i - 1, 3])
                
        if sequences is None:
            return y
        else:
            loss = (loss * sequences_mask).sum() / sequences_mask.sum()
            return y, loss
