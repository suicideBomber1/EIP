## Yashwant Nagarjuna || Batch-1 || Assignment-3

### Initialization

At the beginning of the training we initialize the learnable parameters (weights, biases) with some values. Generally, these values are initialized randomly with a fixed mean and standard deviation. Also, there are several different ways in which we can initialize our parameters taking into consideration of the design of our network.

* Random (default) 

Initializing the weights to zero is a bad guess, since the network doesn't train because the gradients are proportional to the weights. So, ideally we need the weights to be small and not zero. Now, the weights need not be small and also the network will not train fast with small values of weights. But, this is our best guess as we cannot start with other values.

The problem with this initialization is the variance keep growing as the number of inputs are incresed and generally the inputs are of high dimensions in deep networks. So, we need a better way of intializing the parameters.

* Xavier Glorot

Ideally, we want the variance of the input to be same as the variance of the output. If, we intialize the weights randomly from a gaussian distribution with mean 0 and some std, the variance will keep on increasing as the number of inputs increase. So, to make the variance same, we make the adjustment of weights on this condition which leads to Xavier initialization. So, we pick the weights from Gaussian distribution with `mean 0` and `variance 1/N`.

```python
w = (np.random.randn(n)) /sqrt(n)
```

### Activation function

* Sigmoid function

Sigmoid sqaushes the input between 0 and 1. It has a nice interepretation of firing rate of the neuron. But, in practice it is not used due to following reasons. 

1. Saturated neurons:

For larger inputs, sigmoid saturated near 1 and the gradients are zero, which will kill the gradients travelling into the inner layers.

2. Ouputs are not zero centered.

This issue is not of much importance, but the gradients w.r.t 'w' are always positive or always negative.

* ReLU (Rectified Linear Unit)

ReLu function is f(x) = max(0, x) and has been most used in recent years. The advantages of ReLU's are:

1. Computationally very easy to implement
2. Gradients travel very easily because of linear form.

The disadvantages of ReLU are that the neurons can die in the negative regions and can be possible that some neurons of the network are never activated in the training.

* Leaky ReLU

To address the problem in ReLU of the dying neurons, we make the function the negative side not completely zero, but a small linear unit with negative slope. However, the slope in this negative region is a hyper parameter.

* ELU (Exponential Linear Unit)

ELU makes the smooth transition in the negative range and produces closer to zero mean ouputs. 

* SELU

SELU is similar to ELU. For 
x > 0 --> f(x) = lambda * x
x < 0 --> f(x) = lambda * alpha * (exp(x) - 1)

Here 'lambda' and 'alpha' are not hyper parameters but are determined from the inputs.
So, the function looks similar to ELU. Also, the weight intialization is Xavier Init.
With SELU the activations go to mean 0 and variance 1.  This technique beats ReLU + BN in accuracy.

### Regularization

* Dropout
* Label smoothing


