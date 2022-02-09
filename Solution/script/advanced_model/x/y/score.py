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
# On renome "Model" car déjà utilisé par keras
from azureml.core.model import Model as Modelazure

from PIL import Image

from azureml.core.authentication import ServicePrincipalAuthentication


def init():
    global final_model
    global x_val_path
    keras_path = Modelazure.get_model_path(model_name = 'final_model')
    INPUT_SHAPE = (256,256,3)
    NB_CLASS=8
    # Configuration authentification Workspace pour download les images du datastore
    try:
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
    except:
        print("Connexion not working")
        
    # Récupération du datastore
    try:
        datastore = Datastore.get(ws, 'workspaceblobstore')
        datastore.download("./img/")
    except:
        print("Download Datastore not working")
    # Récupération des images téléchargés
    try:
        dir_x="./img/dataset/inputs/"
        _, x_val_path = get_all_file(dir_x)
    except:
        print("File datastore not found")

    try: 
        print(keras_path)
        # Création du modèle final
        mean_iou_metric = Mean_IoU_custom()
        print("Metric loaded")
        final_model = build_vgg16_unet(INPUT_SHAPE,n_classes=NB_CLASS)
        print("model load correctly")
        freeze_layer_vgg16(final_model)
        print("freeze done")
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou_metric])
        # Chargement de la modélisation final
    except:
        print("Load model not working")
        
        
    try:
        final_model.load_weights(str(keras_path) + "/checkpoint")
    except:
        print("Weight model not working")
        

def get_all_file(root_dir,file_pattern = None):
    if root_dir[-1] != "/":
        root_dir +="/"
    train_dir = root_dir + "train/"
    val_dir = root_dir + "val/"
    
    if file_pattern is not None:
        label_id_pattern = file_pattern
    else:
        label_id_pattern = ""
    
#     print(os.listdir("./dataset"),os.listdir("./dataset/oup/"))
    train_set = []
    val_set = []
    
    # Train set
    for dir_ in os.listdir(train_dir):
        for file in os.listdir(train_dir + dir_):
            if label_id_pattern in file:
                train_set.append(train_dir + dir_ +"/"+ file)
    # Validation set
    for dir_ in os.listdir(val_dir):
        for file in os.listdir(val_dir + dir_):
            if label_id_pattern in file:
                val_set.append(val_dir + dir_ +"/"+ file)

    
    return train_set, val_set
        
def run(request):
    # Récupération des informations datastore
    try:
        req = json.loads(request)
        id_img = req["id"]
        img = cv2.imread(x_val_path[int(id_img)])
    except Exception as e:
        error = str(e)
        return error
    # Préprocessing / prédiction / transformation
    try:
        data_transform,size_x,size_y = preprocess_data(img)
        y_pred = final_model.predict(data_transform)
        print("prediction done")
        mask = np.argmax(y_pred,axis=3)
        mask_color = addColors(mask)
        mask_color = resize_img(mask_color,size_x,size_y)
        
        # Merge image avec mask plus orignal img
        mask_color = Image.fromarray(mask_color)
        print("Mask done")
#         mask_color.save('./prediction/predicted_mask.png')
        mask_color.putalpha(120)
        img = Image.open(x_val_path[int(id_img)])
        img.paste(mask_color, (0, 0), mask_color)
        return np.array(img).tolist()
    except Exception as e:
        error = str(e)
        return error
    
def preprocess_data(data):
    org_X = data.shape[0]
    org_Y = data.shape[1]
    img = cv2.resize(data,(256,256))
    img = img /255. 
    if len(data.shape) == 3:
        img = [img]
    img = np.array(img,dtype=np.uint64)
    return np.array(img,dtype=np.uint8),org_X,org_Y

def resize_img(img,org_X,org_Y):
    return cv2.resize(img,(org_Y,org_X),interpolation = cv2.INTER_AREA)
    
def addColors(mask):
    mask = mask[0]
    im = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in range(mask.shape[0]):
         for u in range(mask.shape[1]):
            if mask[i,u]==7:
                    im[i,u]= np.array([0, 0, 255])
            if mask[i,u]==6:
                    im[i,u]= np.array([255, 0, 0])
            if mask[i,u]==5:
                    im[i,u]= np.array([0, 204, 204])
            if mask[i,u]==4:
                    im[i,u]= np.array([0, 255, 0])
            if mask[i,u]==3:
                    im[i,u]= np.array([255, 0, 127])
            if mask[i,u]==2:
                    im[i,u]= np.array([255, 151, 0])
            if mask[i,u]==1:
                    im[i,u]= np.array([153, 153, 0])
            if mask[i,u]==0:
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
