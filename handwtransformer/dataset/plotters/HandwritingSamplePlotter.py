
from typing import List

import numpy as np
from handwtransformer.dataset.HandwritingSample import HandwritingSample
from matplotlib import pyplot as plt

def plot_handwriting_sample(handwriting_sample: HandwritingSample) -> plt.Figure:
    """Plots a handwriting sample.

    Args:
        handwriting_sample (HandwritingSample): The handwriting sample to plot.
    """
    fig = plt.figure(figsize=(20, 5))
    plt.title(handwriting_sample.text)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    for stroke in handwriting_sample.strokes:
        plt.plot(stroke[:, 0], stroke[:, 1], color='black')
        
    return fig
    
def plot_handwriting_samples(handwriting_samples: List[HandwritingSample]) -> plt.Figure:
    """Plots a list of handwriting samples as subfigures.

    Args:
        handwriting_samples (List[HandwritingSample]): The handwriting samples to plot.
    """
    fig = plt.figure(figsize=(20, 5))
    for i, handwriting_sample in enumerate(handwriting_samples):
        plt.subplot(1, len(handwriting_samples), i + 1)
        plt.title(handwriting_sample.text)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        for stroke in handwriting_sample.strokes:
            plt.plot(stroke[:, 0], stroke[:, 1], color='black')
    return fig

def animate_handwriting_sample(handwriting_sample: HandwritingSample) -> plt.Figure:
    """Animates a handwriting sample.

    Args:
        handwriting_sample (HandwritingSample): The handwriting sample to animate.
    """
    import matplotlib.animation as animation
    max_x = max([stroke[:, 0].max() for stroke in handwriting_sample.strokes])
    max_y = max([stroke[:, 1].max() for stroke in handwriting_sample.strokes])
    min_x = min([stroke[:, 0].min() for stroke in handwriting_sample.strokes])
    min_y = min([stroke[:, 1].min() for stroke in handwriting_sample.strokes])
    
    fig = plt.figure(figsize=(20, 5))
    plt.title(handwriting_sample.text)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    lines = []
    for stroke in handwriting_sample.strokes:
        line, = plt.plot([], [], color='black')
        lines.append(line)
    def animate(i):
        ticks_so_far = 0
        for j, stroke in enumerate(handwriting_sample.strokes):
            if i < ticks_so_far + stroke.shape[0]:
                lines[j].set_data(stroke[:i - ticks_so_far, 0], stroke[:i - ticks_so_far, 1])
                break
            ticks_so_far += stroke.shape[0]
    ani = animation.FuncAnimation(fig, animate, frames=range(sum([stroke.shape[0] for stroke in handwriting_sample.strokes])), interval=20)
    return ani
