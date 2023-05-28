import torch

class HandwritingTransformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.letter_embedding = torch.nn.Embedding(256, 256)
        self.positional_encoding = torch.nn.Embedding(1000, 256)
        
    def forward(self, text: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Forward pass of the handwriting transformer.

        Args:
            text (torch.Tensor): The text to generate the handwriting for. Shape == (batch_size, max_text_len)

        Returns:
            torch.Tensor: The generated handwriting. Shape == (batch_size, max_seq_len, 4)
        """
        x = self.letter_embedding(text) # Shape == (batch_size, max_text_len, 256)
        # Add positional encoding
