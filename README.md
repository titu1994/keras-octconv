# Keras Octave Convolutions (OctConv)
-----

Keras implementation of the Octave Convolution blocks from the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049).

<img src="https://github.com/titu1994/keras-octconv/blob/master/images/octconv.png?raw=true" height=100% width=100%>

# Usage
`Octave Convolutions` are a semi-drop-in-replacement for regular convolution layers.

They are implemented in 3 major steps:

## Intiailization of Dual Path Flow
Use the `initial_octconv` block from `octave_conv.py` to initialize the Octave convolution blocks. This function accepts a single input tensor, and returns **two** output tensors : The high frequency pathway and low frequency pathway tensors, in that order

```python

ip = Input(...)

x_high, x_low = initial_conv(ip, ...)
```

## Add any number of Octave Convolution Blocks

Once the two frequency pathways have been obtained, use any number of `octconv_block` from `octave_conv.py` to make the network larger.

**NOTE**:

Each of these blocks accept *two* input tensors, and emits *two* output tensors.

```python

x_high, x_low = octconv_block(x_high, x_low, ...)
x_high, x_low = octconv_block(x_high, x_low, ...)
x_high, x_low = octconv_block(x_high, x_low, ...)

```

## Merging the streams back together

Once you are finished adding `octconv_block`s, merge the two frequency pathways using `final_octconv` from `octave_conv.py`.

This block accepts *two* input tensors and *one* output tensor.

```python

x = final_octconv(x_high, x_low, ...)

...
```


## Acknowledgements
This code is heavily based on the MXNet implementation by [terrychenism](https://github.com/terrychenism) at https://github.com/terrychenism/OctaveConv.

# Requirements
----
- Keras 2.2.4+
- Tensorflow 1.13+ (2.0 support depends on when Keras will support it) / Theano (not tested) / CNTK (not tested)


