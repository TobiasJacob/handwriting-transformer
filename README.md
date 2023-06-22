# Handwriting transformer

Given a sequence of input tokens $x_t$, the goal is to determine the probability of a certain handwriting given the input tokens. It is modeled as a sequence of conditional distributions $P(y | x) = \prod_{i=1}^T P(y_i | x, y_{:i})$, where $P(y_t | x, y_{:t})$ is the probability to move the pen in a certain way. This falls into the domain of behavior cloning or offline reinforcement learning.

In general, want to minimize the log-likelihood of the data $L = - \log P(y | x) = \sum_{i=1}^T - \log P(y_i | x_{i}, y_{:i})$.

$y_t$ is a vector that is ideally $(1, ...)^T$ if a stroke ends, and $(0, d_x, d_y)$ with $d_x, d_y$ being the pen movement if the stroke doesn't end. We model $P(y_t | x, y_{:t})$ as a combination of a bernoulli distribution for the end of stroke, and a gaussian mixture for the position delta. It follows:

- $P(y| x, y_{:t}) = \sum_m \pi_m(NN) N(x_{t+1}|NN) \begin{cases} e \text{ if } y_3 == 0 \\ 1 - e \text{ if } y_3 == 1 \end{cases}$


The mixture density network is trained to minimize the negative log-likelihood of the data $L = - \log P(y | x) = \sum_{i=1}^T - \log P(y_i | x_{i}, y_{:i})$.

## Model

The model is a transformer network with a mixture density network on top. The transformer network is trained to predict the parameters of the mixture density network. The mixture density network is trained to predict the probability of the data.

### Transformer network

The transformer network is a sequence-to-sequence model. It takes a sequence of input tokens $x_t$ and outputs a sequence of output tokens $y_t$. The input tokens are the pen coordinates and the output tokens are the parameters of the mixture density network.

### Mixture density network

The mixture density network is a mixture of Gaussians. It takes the output tokens of the transformer network and outputs the probability of the data.




## Setup

### Download data

Download the offline data from [IAM](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) and extract it to the `data` folder.

Extract the tar.gz files using

```
tar -xzf data/lineStrokes-all.tar.gz -C data/iam
tar -xzf data/ascii-all.tar.gz -C data/iam
```

## Prepare data

```
python -m handwtransformer.prepare
```

## Train

```
python -m handwtransformer
```
