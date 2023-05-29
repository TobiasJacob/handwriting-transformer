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
        # self.final_attention = torch.nn.MultiheadAttention(encoding_size, 8, batch_first=True)
        self.output = torch.nn.Linear(encoding_size, 6)
        
    def forward(self, text: torch.Tensor, text_mask: torch.Tensor, sequences_so_far: torch.Tensor, sequences_so_far_mask: torch.Tensor, train_sequences: Optional[torch.Tensor], train_sequences_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            text (torch.Tensor): Text to be converted to handwriting. Shape == (batch_size, max_text_len)
            text_mask (torch.Tensor): Mask for the text. Shape == (batch_size, max_text_len)
            sequences_so_far (torch.Tensor): Sequences generated so far. Shape == (batch_size, max_seq_len, 4)
            sequences_so_far_mask (torch.Tensor): Mask for the sequences generated so far. Shape == (batch_size, max_seq_len)
            train_sequences (Optional[torch.Tensor]): Ground truth sequences for training purposes. Shape == (batch_size, 1, 4)
            train_sequences_mask (Optional[torch.Tensor]): Mask for the ground truth sequences. Shape == (batch_size, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        x = self.letter_embedding(text) # Shape == (batch_size, max_text_len, encoding_size)
        # Add positional encoding
        x = x + self.letter_positional_embedding(torch.arange(x.shape[1], device=x.device)) # Shape == (batch_size, max_text_len, encoding_size)
        # Apply self attention and feed forward block
        x = self.block1(x, text_mask) # Shape == (batch_size, max_text_len, encoding_size)
        
        command_encodings = self.command_positional_embedding(torch.arange(sequences_so_far.shape[1], device=x.device)) # Shape == (max_text_len, encoding_size)
        sequences_so_far = self.command_embedding(sequences_so_far) # Shape == (batch_size, max_seq_len, encoding_size)
        sequences_so_far = sequences_so_far + command_encodings # Shape == (batch_size, max_seq_len, encoding_size)
        
        input = torch.concat((x, sequences_so_far), dim=1) # Shape == (batch_size, max_text_len + max_seq_len, encoding_size)
        input_mask = torch.concat((text_mask, sequences_so_far_mask), dim=1) # Shape == (batch_size, max_text_len + max_seq_len)

        # Command encodings
        z = self.block2(input, input_mask) # Shape == (batch_size, max_text_len + i, encoding_size)
        z = self.block3(z, input_mask) # Shape == (batch_size, max_text_len + i, encoding_size)
        # z, _ = self.final_attention(torch.ones_like(z[:, 0:1]), z, z) # Shape == (batch_size, 1, encoding_size) # TODO: Figure out a better way to do this
        z = z[:, -1:] # Shape == (batch_size, 1, encoding_size) # TODO: Figure out a better way to do this
        prob_params = self.output(z) # Shape == (batch_size, 6)
        
        pen_delta = torch.distributions.MultivariateNormal(prob_params[:, 0, :2], torch.diag_embed(torch.exp(prob_params[:, 0, 2:4]))) # Event_Shape == (batch_size, 2)
        stroke_end = torch.distributions.Bernoulli(torch.sigmoid(prob_params[:, 0, 4])) # Event_Shape == (batch_size,)
        episode_end = torch.distributions.Bernoulli(torch.sigmoid(prob_params[:, 0, 5])) # Event_Shape == (batch_size,)
        
        # Compute predictions or loss
        if train_sequences is None:
            output = torch.zeros((x.shape[0], 1, 4), device=x.device)
            output[:, 0, :2] = pen_delta.sample() # Shape == (batch_size, max_seq_len, 2)
            output[:, 0, 2] = stroke_end.sample() # Shape == (batch_size, max_seq_len)
            output[:, 0, 3] = episode_end.sample() # Shape == (batch_size, max_seq_len)
            return output
        else:
            loss = torch.zeros((x.shape[0]), device=x.device)
            loss += -pen_delta.log_prob(train_sequences[:, 0, :2])
            loss += -stroke_end.log_prob(train_sequences[:, 0, 2])
            loss += -episode_end.log_prob(train_sequences[:, 0, 3])
            loss = (loss * train_sequences_mask).sum() / train_sequences_mask.sum()
            loss = loss.sum()
            return loss
