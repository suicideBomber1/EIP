# Convolution

Convolution is applying the function operation by striding the window (kernel or filter) across the matrix (generally image).

<p align='center'>
<img width='200' height='200' src="https://raw.githubusercontent.com/mingruimingrui/Convolution-neural-networks-made-easy-with-keras/master/imgs/filtering-many-to-one.gif">
Convolution operation (Source: GitHub)
 </p>


0 0 0 0 ----- 0 0 0 0                                                                                             
0 0 0 0 |255| 0 0 0 0                                                                             
0 0 0 0 |255| 0 0 0 0   
0 0 0 0 |255| 0 0 0 0                                                                              
0 0 0 0 |255| 0 0 0 0   
0 0 0 0 |255| 0 0 0 0                                      
0 0 0 0 |255| 0 0 0 0                                 
0 0 0 0 |255| 0 0 0 0                                                        
0 0 0 0 -----  0 0 0 0                                                          

**Fig.1**  This is `"1"` represented in pixel values (255-white, 0-black)

0 0 0 0-----------0 0 0 0    
0 0 0 |255 255 255| 0 0 0      
0 0 0 |255  0   255| 0 0 0      
0 0 0 |255  0   255| 0 0 0    
0 0 0 |255  0   255| 0 0 0               
0 0 0 |255  0   255| 0 0 0                                                                          
0 0 0 |255  0   255| 0 0 0     
0 0 0 |255 255 255| 0 0 0     
0 0 0 0 -----------0 0 0 0  

**Fig.2**  This is `"0"` represented in pixel values (255-white, 0-black)                                  

As seen in the above gif, we look at a part of a matrix and then `convolution` operation is applied on the sub-part. Consider a 3x3 filter to detect vertical lines in the image. When the convolution is applied on Fig.1 & Fig.2, it returns higher values in there are vertical lines in the image and low values otherwise. Similarly, we can have a filter to detect horizontal lines. If there are two vertical and two horizontal lines detected, we can label the image as "0" and if there is one vertical line detected, we can label the image as "1". Thus, `Convolution` operation can be used to detect features.

### How is `convolution` used in neural networks?
> We know neural networks can learn the parameters of a network by back-propagation through the loss function. We have seen above that `convolution` operation can be used to detect a particular feature in the image. So, combine these two operations by first initializing the required filters (kernels) with some random values (not always true, still an active area of research) and then make the neural network learn the parameters of the filter. So, this concept can be extended to detect different features in images. Ex: Dogs, Cats, etc.

# Feature Map

Feature map (Activation map) is the output activation of the given filter. Each feature map will detect a feature (or a combination of features) and the location from the original input. Feature maps from different filters are stacked on one another and the process is continued depending on the architecture of the model also the extent of differentiation we expect for the input.

<p align='center'>
<img width='300' height='300' src="https://adeshpande3.github.io/assets/ActivationMap.png">
Source: Michael Nielsen
 </p>


# Receptive Field

When dealing with higher order input such as images it is not possible to connect all neurons with each other. Instead we connect each neuron to a local region of input space. 

Receptive field is the region of `input space` that a particular feature is looking. Every neuron in conv layer represents a filter applied to previous layer. The area of the previous layer that this filter is applied to is called *receptive field* of that layer. This receptive field increases as we stack more conv layers.

![Diagram showing the receptive field of conv layer](https://qph.ec.quoracdn.net/main-qimg-34686eb9aa41d84ec784164601174be5.webp)

A, B, C are three conv layers extracting one feature each. (Stride is 1, There is padding not shown in fig, filter size is 3x3)

The receptive field of B (2, 2) is through A (1, 1) and A (3, 3), as this element is formed from convolution of filter with (1, 1) through (3, 3) region of A. Therefore, the receptive field of a neuron in B in 3x3.

Now, consider a neuron in C say (3, 3). This neuron is formed from B (2, 2) through B (4, 4), which inturn is formed from the entire A matrix. Therefore, the receptive layer of a neuron in C is 5x5. Hence, the receptive field increases as we stack more and more convolution layers.


