from dataclasses import dataclass
from typing import List

from handwtransformer.dataset.HandwritingSample import HandwritingSample

@dataclass
class Dataset:
    samples: List[HandwritingSample]
