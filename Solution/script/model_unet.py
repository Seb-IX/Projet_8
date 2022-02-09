from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50



def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model

def build_vgg16_unet(input_shape,n_classes,weights="imagenet"):
    
    inputs = Input(input_shape)
    vgg16 = VGG16(include_top=False, weights=weights, input_tensor=inputs)

    # Encoder 
    s1 = vgg16.get_layer("block1_conv2").output         
    s2 = vgg16.get_layer("block2_conv2").output         
    s3 = vgg16.get_layer("block3_conv3").output         
    s4 = vgg16.get_layer("block4_conv3").output         

    # Bridge 
    b1 = vgg16.get_layer("block5_conv3").output          

    # Decoder 
    d1 = decoder_block(b1, s4, 512)                     
    d2 = decoder_block(d1, s3, 256)                     
    d3 = decoder_block(d2, s2, 128)                     
    d4 = decoder_block(d3, s1, 64)                      

    # Output 
    if n_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model

def freeze_layer_vgg16(model):
    i=0
    all_layer_not_trainable = [j for j in range(1,15)] # toutes les couche VGG16
    for layer in model.layers:
        if i in all_layer_not_trainable:
            layer.trainable = False
            print('freeze layer:',layer.name)
        i+=1

def build_resnet50_unet(input_shape,n_classes,weights="imagenet"):

    inputs = Input(input_shape)
    resnet50 = ResNet50(include_top=False, weights=weights, input_tensor=inputs)

    # Encoder
    s1 = resnet50.layers[0].output           
    s2 = resnet50.get_layer("conv1_relu").output        
    s3 = resnet50.get_layer("conv2_block3_out").output  
    s4 = resnet50.get_layer("conv3_block4_out").output  

    # Bridge 
    b1 = resnet50.get_layer("conv4_block6_out").output  

    # Decoder 
    d1 = decoder_block(b1, s4, 512)                     
    d2 = decoder_block(d1, s3, 256)                     
    d3 = decoder_block(d2, s2, 128)                     
    d4 = decoder_block(d3, s1, 64)                      

    # Output
    if n_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model


def freeze_layer_resnet50(model):
    i=0
    all_layer_not_trainable = [j for j in range(0,123)] # toutes les couche ResNet50
    for layer in model.layers:
        if i in all_layer_not_trainable:
            layer.trainable = False
        i+=1
        