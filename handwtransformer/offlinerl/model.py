import torch

class HandwritingTransformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
