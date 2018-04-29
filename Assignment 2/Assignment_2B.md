## Yashwant Nagarjuna | Batch-1

### Assignment 2A
[GitHub link](https://github.com/suicideBomber1/EIP/blob/master/Assignment%202/Yashwant_Nagarjuna_Assignment_2A.ipynb)

### Assignment 2B 

[Jupyter notebook](https://github.com/suicideBomber1/EIP/blob/master/Assignment%202/Assignment_2B.ipynb)

**Step 0**: Read input and output
```python
X = np.array([[1, 0, 1, 0], 
            [1, 0, 1, 1],
            [0, 1, 0, 1]])
```
**Step 1**: Initialize weights and biases with random values (There are methods to initialize weights and biases but for now initialize with random values)

```python
wh = np.random.randn(4, 3)
bh = np.random.randn(1, 3)
wout = np.random.randn(3, 1)
bout = np.random.randn(1,)
```
**Step 2**: Calculate hidden layer input:

```python
hidden_layer_input = np.dot(X, wh) + bh
```
**Step 3**: Perform non-linear transformation on hidden linear input
```python
hidden_layer_activations = sigmoid(hidden_layer_input)
```

**Step 4**: Perform linear and non-linear transformation of hidden layer activation at output layer

```python
output_layer_input = np.dot(hidden_layer_activations, wout) + bout
```

**Step 5**: Calculate gradient of Error(E) at output layer
```python
E = y - output
```

**Step 6**: Compute slope at output and hidden layer
```python
slope_output_layer = (1-sigmoid(output_layer_input))
slope_hidden_layer = (1-sigmoid(hidden_layer_input))
```

**Step 7**: Compute delta at output layer
```python
d_output = E*slope_output_layer*lr
```

**Step 8**: Calculate Error at hidden layer
```python
error_at_hidden_layer = np.dot(d_output, wout.T)
```

**Step 9**: Compute delta at hidden layer
```python
d_hidden_layer = error_at_hidden_layer*slope_hidden_layer
```

**Step 10**: Update weight at both output and hidden layer
```python
wout += np.dot(hidden_layer_activations, d_output)*lr
wh += np.dot(X.T, d_hidden_layer)*lr
```

**Step 11**: Update biases at both output and hidden layer
```python
bout += np.sum(d_output, axis=0)*lr
bh += np.sum(d_hidden_layer, axis=0)*lr
```

![Boom ](http://gph.is/2cxuYuL)







