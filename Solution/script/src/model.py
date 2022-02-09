import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K

from keras.models import Model

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate

from tensorflow.keras.applications import VGG16

import cv2


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
        

class GeneratorCitySpace(Sequence):
        
    CATS = {
        'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
    }
    
    def _convert_mask(self,img):
        img = np.squeeze(img)
        mask = np.zeros((img.shape[0], img.shape[1], 8),dtype=np.uint8)
        for i in range(-1, 34):
            if i in self.CATS['void']:
                mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
            elif i in self.CATS['flat']:
                mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
            elif i in self.CATS['construction']:
                mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
            elif i in self.CATS['object']:
                mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
            elif i in self.CATS['nature']:
                mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
            elif i in self.CATS['sky']:
                mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
            elif i in self.CATS['human']:
                mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
            elif i in self.CATS['vehicle']:
                mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
        return np.array(mask,dtype='uint8')
    
    def _transform_data(self,X,Y):
        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis = 3)
        X = X /255. 
        return np.array(X,dtype=np.uint8), Y
    
    def __init__(self, image_filenames, labels, batch_size,crop_x,crop_y):
        """Générateur de données avec augmentation des images
        """
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.crop_x,self.crop_y = crop_x, crop_y

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x=[cv2.resize(cv2.imread(path_X),(self.crop_x,self.crop_y)) for path_X in batch_x]
        y = [cv2.resize(self._convert_mask(cv2.imread(path_Y,0)),(self.crop_x,self.crop_y)) for path_Y in batch_y]
        y=np.array(y)
        x=np.array(x)
        return self._transform_data(x,y)
    
class Mean_IoU_custom(tf.keras.metrics.Metric):
    """Metrique tensorflow modifié pour être utiliser avec un masque sur une modélisation de segmentation sémantique 
    
arguments:
- n_class : correspond au nombre de classe du mask (ou le nombre de channel du mask final)
    
    """
    def __init__(self,name='mean_iou', n_class=8, **kwargs):
        super(Mean_IoU_custom, self).__init__(name=name, dtype=None)
        self.n_class=n_class
        self.result_assign = self.add_weight('result',shape=(),initializer="zeros")

    def update_state(self, y_true, y_pred,sample_weight=None):
        if len(y_true.shape) == 4:
            y_pred = tf.argmax(y_pred,axis=3)
            y_true = tf.argmax(y_true,axis=3)
        mean_iou = 0.0
        seen_classes = 0.0
        # K.switch permet d'appliquer une condition ternaire 
        # K.equal permet de vérifier que les 2 Tensor sont égaux
        # K.cast permet de convertir un tensor en différent type (ici on utilise principalement les dtype float
        # K.sum permet d'obtenir la sum des valeurs du Tensor (matrice/vecteur)
        for c in range(self.n_class):
            labels_c = K.cast(K.equal(y_true, c), K.floatx())
            pred_c = K.cast(K.equal(y_pred, c), K.floatx())

            labels_c_sum = K.sum(labels_c)
            pred_c_sum = K.sum(pred_c)
            
            intersect = K.sum(labels_c*pred_c)
            union = labels_c_sum + pred_c_sum - intersect
            iou = intersect / union
            condition = K.equal(union, 0)
            mean_iou = K.switch(condition,mean_iou,mean_iou+iou)
            seen_classes = K.switch(condition,seen_classes,seen_classes+1)
            
        self.result_assign = K.switch(K.equal(seen_classes, 0),mean_iou,mean_iou/seen_classes)
        
    def reset_state(self):
        self.result_assign = 0
    
    def reset_states(self):
        self.result_assign = 0
    
    def result(self):
        return self.result_assign    
