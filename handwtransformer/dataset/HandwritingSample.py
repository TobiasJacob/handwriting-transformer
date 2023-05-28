
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class HandwritingSample:
    text: str
    strokes: List[np.ndarray] # List[np.ndarray[shape==(2,line_length)]] describing a list of strokes, each stroke having multiple 2d points
