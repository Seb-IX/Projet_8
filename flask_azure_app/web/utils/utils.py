import keras
import tensorflow
from function.metrics import Mean_IoU_custom

import json
import os
import re
import numpy as np
import cv2

import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16


from keras.models import Sequential, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.models import Model

from azureml.core import Datastore, Workspace
from azureml.core import Model as Modelazure

from azureml.core.authentication import ServicePrincipalAuthentication
# from azureml.core.model import Model

from PIL import Image


IMAGE_INPUT_PATH = "./static/data/img/"
PATH_MODEL_AZURE = "./static/data/model_azure/"


def create_model(INPUT_SHAPE = (256,256,3),NB_CLASS=8):
    mean_iou_metric = Mean_IoU_custom()
    final_model = build_vgg16_unet(INPUT_SHAPE,n_classes=NB_CLASS)
    freeze_layer_vgg16(final_model)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou_metric])
    return final_model

def load_model_azure():
    ## Connexion à Azure et récupération du modèle final enregistrer
    subscription_id = os.environ["SUBSCRIPTION_ML"]
    resource_group = os.environ["RESOURCE_GROUP_ML"]
    workspace_name = os.environ["WORKSPACE_ML"]
    
    svc_pr_password = os.environ.get("AZUREML_PASSWORD") # Durée 6 mois
    tenant_id = os.environ.get("TENANT_ID") # ID de l'annuaire (locataire)
    service_principal_id = os.environ.get("SERVICE_PRINCIPAL_ID") # ID de l'application

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=service_principal_id,
        service_principal_password=svc_pr_password)
    
    ws = Workspace(subscription_id=subscription_id,
               resource_group=resource_group,
               workspace_name=workspace_name,  
               auth=svc_pr
    )

    # weight_model = Modelazure(ws, 'final_model')
    weight_model = Modelazure(ws, 'final_model')
    try: 
        weight_model.download("./static/data/model_azure/", exist_ok=True)
    except:
        print("not working download()")

    final_model = create_model()
    # Chargement de la modélisation final
    try:
        final_model.load_weights("./static/data/model_azure/final_model/checkpoint")
    except:
        final_model.load_weights("./static/data/model_azure/checkpoint")


    return final_model

def load_model():
    final_model = create_model()
    # Chargement de la modélisation final poids local
    final_model.load_weights('./static/data/model_vgg16_unet_no_augmented/checkpoint')

    return final_model

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

def prepare_data(data):
    response = {'success': True}
    data.save("./static/data/prediction/predicted_mask_on_img.png")
    response["prediction"]="/static/data/prediction/predicted_mask_on_img.png"
    return response

def preprocess_data(id_img):
    if int(id_img) > 9:
        name_image = "berlin_0000" + str(id_img) + "_000019_leftImg8bit.png"
    else:
        name_image = "berlin_00000" + str(id_img) + "_000019_leftImg8bit.png"

    path_image = IMAGE_INPUT_PATH + name_image
    print(path_image)
    data = cv2.imread(path_image)
    print(data.shape)
    org_X = data.shape[0]
    org_Y = data.shape[1]
    img = cv2.resize(data,(256,256))
    img = img /255. 
    if len(data.shape) == 3:
        img = [img]
    img = np.array(img,dtype=np.uint64)
    print(img.shape)
    return np.array(img,dtype=np.uint8),org_X,org_Y,path_image

def resize_img(img,org_X,org_Y):
    return cv2.resize(img,(org_Y,org_X),interpolation = cv2.INTER_AREA)
    
def addColors(mask):
    mask = mask[0]
    im = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in range(mask.shape[0]):
         for u in range(mask.shape[1]):
            if mask[i,u]==7: # vehicle
                    im[i,u]= np.array([0, 0, 255])
            if mask[i,u]==6: # human
                    im[i,u]= np.array([255, 0, 0])
            if mask[i,u]==5: # sky
                    im[i,u]= np.array([0, 204, 204])
            if mask[i,u]==4: # nature
                    im[i,u]= np.array([0, 255, 0])
            if mask[i,u]==3: # object
                    im[i,u]= np.array([133, 0, 121])
            if mask[i,u]==2: # construction
                    im[i,u]= np.array([255, 151, 0]) 
            if mask[i,u]==1: # flat
                    im[i,u]= np.array([161, 161, 161]) 
            if mask[i,u]==0: # void
                    im[i,u]= np.array([0, 0, 0])
    return im

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



