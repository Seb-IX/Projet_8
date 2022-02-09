import cv2
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K


class DatasetLoader(Sequence):
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
        return np.array(np.argmax(mask,axis=2), dtype='uint8')
        
    def _augmented_image(self,path_X, path_Y,batch_size):
        image=cv2.imread(path_X)
        mask = self._convert_mask(cv2.imread(path_Y,0))
        img_ = cv2.resize(image,(self.crop_x,self.crop_y))
        mask_ = cv2.resize(mask,(self.crop_x,self.crop_y))
        segmap = SegmentationMapsOnImage(mask_, shape=img_.shape)
        images = [np.copy(img_) for _ in range(batch_size)]
        segmaps = [segmap for _ in range(batch_size)]
        batches = [UnnormalizedBatch(images=images,segmentation_maps=segmaps) for _ in range(1)]
        ia.seed(2)
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10)), # fast permet de faire une rotation sur l'image
            iaa.GammaContrast((0.5, 2.0)), # very fast, augmente ou réduit le constraste 
            iaa.Fliplr(0.5),  # fast, retourne l'image
            iaa.CropAndPad(px=(-10, 10)),  # very fast, permet de rogner un peu l'image pour la décaller
            iaa.Salt((0.0,0.05)) # very fast, rajoute du bruit
        ])
        batches_aug = list(seq.augment_batches(batches, background=False))
        final_img = [batches_aug[0].images_aug[i] for i in range(batch_size)]
        final_mask = [batches_aug[0].segmentation_maps_aug[i].get_arr() for i in range(batch_size)]
        final_img.append(img_)
        final_mask.append(mask_)
        return final_img,final_mask
    
    def _get_images(self,path_X, path_Y):
        image=cv2.imread(path_X)
        mask = self._convert_mask(cv2.imread(path_Y,0))
        img_ = cv2.resize(image,(self.crop_x,self.crop_y))
        mask_ = cv2.resize(mask,(self.crop_x,self.crop_y))
        return img_,mask_
    
    def _prepared_data(self,path_X_list,path_Y_list):
        x = []
        y = []
        for path_x,path_y in zip(path_X_list,path_Y_list):
            if self.data_augmented > 0:
                img_,mask_ = self._augmented_image(path_x,path_y,batch_size=self.data_augmented)
                x.extend(img_)
                y.extend(mask_)
            else:
                img_,mask_ = self._get_images(path_x,path_y)
                x.append(img_)
                y.append(mask_)
        x,y = np.array(x,dtype=np.uint8),np.array(y,dtype=np.uint8)
        y = np.expand_dims(y, axis = 3)
        x = np.array(x /255.,dtype=np.uint8)
        train_masks_cat = to_categorical(y, num_classes=8)
        y = train_masks_cat.reshape((y.shape[0], y.shape[1], y.shape[2], 8))
        return x,y
        
    def __init__(self,x_list,y_list,crop_x,crop_y,batch_size,data_augmented=8):
        """
        Générateur de données avec augmentation des images
        """
        self.x_list = x_list
        self.y_list = y_list
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.batch_size = batch_size
        self.data_augmented = data_augmented
        
    def __len__(self):
        return int(np.ceil(len(self.x_list) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.x_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        x,y= self._prepared_data(batch_x,batch_y)
        return x,y
    
    
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
    
class Dice_coefficient(tf.keras.metrics.Metric):
    """
    Metrique tensorflow modifié pour être utiliser avec un masque sur une modélisation de segmentation sémantique. Cette métrique permet de calculer le score de Dice ou dice coef.
    
params:
n_class : correspond au nombre de classe du mask (ou le nombre de channel du mask final)
    
    """
    def __init__(self,name='dice_coef', n_class=8, **kwargs):
        super(Dice_coefficient, self).__init__(name=name, dtype=None)
        self.n_class=n_class
        self.result_assign = self.add_weight('result',shape=(),initializer="zeros")

    def update_state(self, y_true, y_pred,sample_weight=None):
        
        y_true_f = K.cast(K.flatten(y_true), K.floatx())
        y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
        intersection = K.sum(y_true_f * y_pred_f)
        smooth = float(self.n_class)
        self.result_assign = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
#         self.result_assign = (2. * intersection + smooth) / (union + float(self.n_class))
        
    def reset_state(self):
        self.result_assign = 0
    
    def reset_states(self):
        self.result_assign = 0
    
    def result(self):
        return self.result_assign

############# A VERIFIER : ############


def dice_coeff(y_true, y_pred):
    """Cette fonction permet le calcul du score de Dice ou Dice coeff ici on l'utilise avec la fonction dice loss (à améliorer)
    
arguments:
- y_true (type tensorflow.Tensor / numpy.array) : vecteur/matrice des valeurs attendus
- y_pred (type tensorflow.Tensor / numpy.array) : vecteur/matrice des valeurs de prédiction

return
- score (type Tensor/float): retourne le score du coefficient Dice
    """
    smooth = 8.
    y_true_f = K.cast(K.flatten(y_true), K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
    
def dice_loss(y_true, y_pred):
    """Dice loss est une fonction de coût qui se base sur le dice_coeff qui est un dérivé du Mean IoU.
    
arguments:
- y_true (type tensorflow.Tensor / numpy.array) : vecteur/matrice des valeurs attendus
- y_pred (type tensorflow.Tensor / numpy.array) : vecteur/matrice des valeurs de prédiction

return:
- loss (type float): retourne le score de perte basé sur le coefficient Dice.
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    """
    """
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss