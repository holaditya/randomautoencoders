import tensorflow as tf
import layers
from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    conv1 = layers.conv(input, name = 'conv1', filter_dims=[3,3,1], stride_dims=[2,2])
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    conv2 = layers.conv(conv1, name = 'conv2', filter_dims=[3,3,8], stride_dims=[2,2])
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    conv3 = layers.conv(conv2, name = 'conv3', filter_dims=[3,3,8], stride_dims=[2,2])
    # FC: output_dim: 100, no non-linearity
    fc = layers.fc(conv3, name='fc', out_dim = 100)
    return fc
    raise NotImplementedError

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    fc = layers.fc(input,name='dfc',out_dim=128)    
    dfc=tf.reshape(fc, [-1, 4, 4, 8])
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    dconv1=layers.deconv(dfc,name='deconv1',filter_dims=[3,3,8],stride_dims=[2,2],padding='SAME')
    print("dconv1 shape", dconv1.get_shape().as_list())
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    dconv2=layers.deconv(dconv1,name = 'deconv2',filter_dims=[8,8,1],stride_dims= [2,2],padding='VALID')
    print("dconv2 shape", dconv2.get_shape().as_list())
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    dconv3=layers.deconv(dconv2,name='deconv3',filter_dims=[7,7,1],stride_dims=[1,1],padding='VALID',non_linear_fn=tf.nn.sigmoid)
    print("dconv3 shape", dconv3.get_shape().as_list())
    return dconv3
    raise NotImplementedError

def autoencoder(input_shape):
    # Define place holder with input shape
    batch_size, input_h, input_w, num_channels =  input_shape 
    X = tf.placeholder(tf.float32, [batch_size, input_h, input_w, num_channels], name='X_placeholder')
    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(X)
        # Pass encoding into decoder to obtain reconstructed image
        decoding = decoder(encoding)
        # Return input image (placeholder) and reconstructed image
        return X, decoding
        pass
