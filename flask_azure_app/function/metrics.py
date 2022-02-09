
import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers


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