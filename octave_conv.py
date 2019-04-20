from keras.layers import Conv2D, AveragePooling2D, UpSampling2D
from keras.layers import add


def initial_octconv(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                    alpha=0.5, padding='same', dilation=None, bias=False):
    """
    Initializes the Octave Convolution architecture.

    Accepts a single input tensor, and returns a pair of tensors.
    The first tensor is the high frequency pathway.
    The second tensor is the low frequency pathway.

    # Arguments:
        ip: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        alpha: float between [0, 1]. Defines the ratio of filters
            allocated to the high frequency and low frequency
            branches of the octave conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.

    # Returns:
        a pair of tensors:
            - x_high: high frequency pathway.
            - x_low: low frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    high_low_filters = int(alpha * filters)
    high_high_filters = filters - high_low_filters

    if strides[0] > 1:
        ip = AveragePooling2D()(ip)

    # High path
    x_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                    dilation_rate=dilation, use_bias=bias,
                    kernel_initializer='he_normal')(ip)

    # Low path
    x_high_low = AveragePooling2D()(ip)
    x_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                   dilation_rate=dilation, use_bias=bias,
                   kernel_initializer='he_normal')(x_high_low)

    return x_high, x_low


def final_octconv(ip_high, ip_low, filters, kernel_size=(3, 3), strides=(1, 1),
                  padding='same', dilation=None, bias=False):
    """
    Ends the Octave Convolution architecture.

    Accepts two input tensors, and returns a single output tensor.
    The first input tensor is the high frequency pathway.
    The second input tensor is the low frequency pathway.

    # Arguments:
        ip_high: keras tensor.
        ip_low: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.

    # Returns:
        a single Keras tensor:
            - x_high: The merged high frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    if strides[0] > 1:
        avg_pool = AveragePooling2D()

        ip_high = avg_pool(ip_high)
        ip_low = avg_pool(ip_low)

    # High path
    x_high_high = Conv2D(filters, kernel_size, padding=padding,
                         dilation_rate=dilation, use_bias=bias,
                         kernel_initializer='he_normal')(ip_high)

    # Low path
    x_low_high = Conv2D(filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(ip_low)

    x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

    # Merge paths
    x = add([x_high_high, x_low_high])

    return x


def octconv_block(ip_high, ip_low, filters, kernel_size=(3, 3), strides=(1, 1),
                  alpha=0.5, padding='same', dilation=None, bias=False):
    """
    Constructs an Octave Convolution block.

    Accepts a pair of input tensors, and returns a pair of tensors.
    The first tensor is the high frequency pathway for both ip/op.
    The second tensor is the low frequency pathway for both ip/op.

    # Arguments:
        ip_high: keras tensor.
        ip_low: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        alpha: float between [0, 1]. Defines the ratio of filters
            allocated to the high frequency and low frequency
            branches of the octave conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.

    # Returns:
        a pair of tensors:
            - x_high: high frequency pathway.
            - x_low: low frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    low_low_filters = high_low_filters = int(alpha * filters)
    high_high_filters = low_high_filters = filters - low_low_filters

    avg_pool = AveragePooling2D()

    if strides[0] > 1:
        ip_high = avg_pool(ip_high)
        ip_low = avg_pool(ip_low)

    # High path
    x_high_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                         dilation_rate=dilation, use_bias=bias,
                         kernel_initializer='he_normal')(ip_high)

    x_low_high = Conv2D(low_high_filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(ip_low)
    x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

    # Low path
    x_low_low = Conv2D(low_low_filters, kernel_size, padding=padding,
                       dilation_rate=dilation, use_bias=bias,
                       kernel_initializer='he_normal')(ip_low)

    x_high_low = avg_pool(ip_high)
    x_high_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(x_high_low)

    # Merge paths
    x_high = add([x_high_high, x_low_high])
    x_low = add([x_low_low, x_high_low])

    return x_high, x_low


if __name__ == '__main__':

    visualize_model = True

    from keras.models import Model
    from keras.layers import Input

    ip = Input(shape=(32, 32, 3))

    xh, xl = initial_octconv(ip, filters=48)

    xh, xl = octconv_block(xh, xl, filters=64)
    xh, xl = octconv_block(xh, xl, filters=96, strides=(2, 2))

    x = final_octconv(xh, xl, filters=10)

    model = Model(ip, x)
    model.summary()

    if visualize_model:
        from keras.utils.vis_utils import plot_model

        plot_model(model, to_file='octconv.png', show_shapes=False)