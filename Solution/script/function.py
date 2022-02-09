import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

# Catégorie récuperer sur https://github.com/fregu856/segmentation/blob/master/preprocess_data.py
cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
}

def check_file_contains_same(x,y):
    """Vérifie que le jeu de données contient autant de donnée d'une part et d'autre

argument: 
- x (type list) :liste des chemins d'accès
- y (type list) :liste des chemins d'accès
    """
    assert len(x) == len(y)

def convert_mask(img,one_hot_encoder=False):
    """Cette méthode permet de convertir l'image '_labelids.png' du jeu de données de CityScapes.
La méthode permet de récupérer l'image au format one_hot_encoder ou au format label_encoder.
arguments:
- img (type numpy.array): image du jeu de données CityScapes '...labelids.png' au format numpy  
- one_hot_encoder (type bool optionnel)(default False) : Permet de choisir le mode de conversion (one_hot_encoder 8 channel ou label_encoder 1 channel)
    
return:
mask (type numpy.array) : mask pour la segmentation sémantique au format one hot encoder ou label encoder avec les 8 catégories principal (void, flat, construction, object, nature, sky, human, vehicle)
    """
    if len(img.shape) == 3:
        img = np.squeeze(img[:,:,0])
    else:
        img = np.squeeze(img)
    mask = np.zeros((img.shape[0], img.shape[1], 8),dtype=np.uint16)
    for i in range(-1, 34):
        if i in cats['void']:
            mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
        elif i in cats['flat']:
            mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
        elif i in cats['construction']:
            mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
        elif i in cats['object']:
            mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
        elif i in cats['nature']:
            mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
        elif i in cats['sky']:
            mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
        elif i in cats['human']:
            mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
        elif i in cats['vehicle']:
            mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
    
    if one_hot_encoder:
        return np.array(mask, dtype='uint8')
    else:
        return np.array(np.argmax(mask,axis=2), dtype='uint8')
        

def get_all_file(root_dir,file_pattern = None):
    """Cette fonction permet de récupérer les données du jeu de données CityScapes, il suffit d'indiquer le répertoire parent avant les dossier train/val/test et de spécifier le pattern du fichier si on récupére les mask (par exemple '_labelIds.png') OU mettre à None si on souhaite récupérer tout les fichiers 
    
argument:
- root_dir (type str): chemin d'accès complet ou partiel au dossier du jeu de données (ex: './data/gtFine/')
- file_pattern (type str optionnel)(default None): indique le pattern recherche pour la récupération des fichiers

return:
train, val, test (type list): return les chemin d'accès des images situer dans chacun des chemins respectif
    """
    if root_dir[-1] != "/":
        root_dir +="/"
    train_dir = root_dir + "train/"
    val_dir = root_dir + "val/"
    test_dir = root_dir + "test/"
    
    if file_pattern is not None:
        label_id_pattern = file_pattern
    else:
#         label_id_pattern = "_labelIds.png"
        label_id_pattern = ""
    
    assert os.path.isdir(train_dir) & os.path.isdir(val_dir) & os.path.isdir(test_dir)
    train_set = []
    val_set = []
    test_set = []
    
    start_get_path = time.time()
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
    # Test set
    for dir_ in os.listdir(test_dir):
        for file in os.listdir(test_dir + dir_):
            if label_id_pattern in file:
                test_set.append(test_dir + dir_ +"/"+ file)
    full_time_get_path = time.time() - start_get_path
    print("Temp pour récupérer les paths : %.2fs" % (full_time_get_path))
    
    return train_set, val_set, test_set

def get_image_labelid(path:str,one_hot_encoder:bool=False):
    """Récupére le mask parfaitement formaté.
    
argument:
- path (type str): chemin d'accès complet ou partiel de l'image "_labelIds.png"
- one_hot_encoder (type bool optionnel) (default: False): Permet de choisir le mode de conversion (one_hot_encoder 8 channel ou label_encoder 1 channel)
    
    return:
    """
    return convert_mask(cv2.imread(path,0),one_hot_encoder=one_hot_encoder)


def vis_mask(photo,masque,cmap_mask = "viridis",figsize=(20,20)):
    """Cette méthode permet de visualiser les mask en séparant les différente couche du one hot encoder
    
argument:
- photo (type numpy.array) : Image d'origine en RGB au format numpy
- masque (type numpy.array, shape: (x,y,8)) : Masque au format one hot encoder 
- cmap_mask (type str optionnel)(default viridis) : permet de choisir le style/couleur d'affichage des masques
- figsize (type tulpe optionnel)(default (20,20)) : permet de choisir la taille de la figure
    
    """
    fig = plt.figure(figsize=figsize)
    
    ax4 = fig.add_subplot(2, 4, 1)
    ax4.set_title('photo original')
    ax4.imshow(photo.squeeze(),filterrad=4.0)

    ax = fig.add_subplot(2, 4, 2)
    ax.set_title('void')
    ax.imshow(masque[...,0].squeeze(),cmap=cmap_mask)

    ax1 = fig.add_subplot(2, 4, 3)
    ax1.set_title('flat')
    ax1.imshow(masque[...,1].squeeze(),cmap=cmap_mask)
    
    ax3 = fig.add_subplot(2, 4, 4)
    ax3.set_title('object')
    ax3.imshow(masque[...,3].squeeze(),cmap=cmap_mask)
    
    ax7 = fig.add_subplot(1, 5, 1)
    ax7.set_title('nature')
    ax7.imshow(masque[...,4].squeeze(),cmap=cmap_mask)

    ax8 = fig.add_subplot(1, 5, 2)
    ax8.set_title('sky')
    ax8.imshow(masque[...,5].squeeze(),cmap=cmap_mask)
    
    ax9 = fig.add_subplot(1, 5, 3)
    ax9.set_title('human')
    ax9.imshow(masque[...,6].squeeze(),cmap=cmap_mask)
    
    ax10 = fig.add_subplot(1, 5, 4)
    ax10.set_title('vehicule')
    ax10.imshow(masque[...,7].squeeze(),cmap=cmap_mask)

    ax2 = fig.add_subplot(1, 5, 5)
    ax2.set_title('construction')
    ax2.imshow(masque[...,2].squeeze(),cmap=cmap_mask)
    
    
def gabor_filter(img,df):
    """Méthode pour la fonction 'prepared_data_linear_model'.
    """
    num = 1 
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):   
                    gabor_label = 'Gabor' + str(num) 
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img 
                    num += 1 

def other_filter(img,df):
    """Méthode pour la fonction 'prepared_data_linear_model'.
    """
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    #GAUSSIAN with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
def prepared_data_linear_model(X_path,Y_path,crop_x=256,crop_y=256):
    """Cette fonction permet de formater une image pour facilité l'apprentissage par un modèle linéaire grâce à l'application de différent filtre (gabor, sobel, roberts, prewitt, gaussian filtre...).
    
arguments:
- X_path (type str): chemin d'accès complet ou partiel de l'image d'entrée
- Y_path (type str): chemin d'accès complet ou partiel de l'image de sortie (masque)
- crop_x (type int optionnel)(default: 256): permet de réduire l'image sur l'axe X afin de réduire les temps de calcul
- crop_y (type int optionnel)(default: 256): permet de réduire l'image sur l'axe Y afin de réduire les temps de calcul

return:
- X,Y (type pandas.DataFrame): retourne en X l'image applatie avec les différent filtre et leur valeur pour chaque pixel, et en Y le label du masque
    """
    final_df = pd.DataFrame()
    for i in range(len(X_path)):
        # Image
        img = cv2.imread(X_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(crop_x,crop_y))
        img2 = img.reshape(-1)
        df = pd.DataFrame()
        df['Original Image'] = img2
        # Création des résultat filter (gabor, sobel, roberts, prewitt, filtre gaussian...)
        gabor_filter(img,df)
        other_filter(img,df)
        # Mask
        labeled_img = get_image_labelid(Y_path[i],one_hot_encoder=False)
        labeled_img = cv2.resize(labeled_img,(crop_x,crop_y))
        labeled_img1 = labeled_img.reshape(-1)
        df['Labels'] = labeled_img1
        if final_df.shape[0] == 0:
            final_df = df.copy()
        else:
            final_df = final_df.append(df.copy(),ignore_index=True)
        
    # Prepared X,Y    
    Y = final_df["Labels"].values
    X = final_df.drop(labels = ["Labels"], axis=1) 
    return X,Y

def create_checkpoint_path(root_dir,name):
    """Cette fonction permet de créer le répertoire du fichier checkpoint
en créant le répertoire si il n'existe pas et retourne l'emplacement final du fichier checkpoint.
Il est impossible d'avoir un nom plus grand de 50 caractère et les espace seront remplacer pour mettre des '_' à la place.
    
arguments:
- root_dir (type str):
- name (type str):

return:
- checkpoint_filepath (type str) : 
    """
    # On limite à 50 caractère max, on retire les espaces et on formate pour bien créer les dossiers
    name = name[:50]
    name = name.replace(" ","_")
    if not root_dir[-1] == "/":
        root_dir+="/"
        
    if name[0] == "/":
        name=name[1:]
        
    if not name[-1] =="/":
        name = name + "/"
        
    checkpoint_dirpath = root_dir + name
    checkpoint_filepath = checkpoint_dirpath + 'checkpoint'
    os.makedirs(checkpoint_dirpath, exist_ok=os.path.exists(checkpoint_dirpath))
    return checkpoint_filepath

def create_checkpoint_callback(root_dir :str,name: str,metric):
    """Cette fonction permet de créer le répertoire du fichier checkpoint
en créant le répertoire si il n'existe pas et retourne l'emplacement final du fichier checkpoint et l'object callback utiliser par Tensorflow.
Il est impossible d'avoir un nom plus grand de 50 caractère et les espace seront remplacer pour mettre des '_' à la place.
    
arguments:
- root_dir (type str):
- name (type str):
- metric (type tensorflow.keras.metric.Metrics):

return:
- model_checkpoint_callback_gen (type tensorflow.keras.callbacks.ModelCheckpoint): 
- checkpoint_filepath (type str) : 
    """
    checkpoint_filepath = create_checkpoint_path(root_dir,name)
    monitor_get = "val_"+ metric.name
    
    model_checkpoint_callback_gen = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor_get,
        mode='max',
        save_best_only=True)
    return model_checkpoint_callback_gen, checkpoint_filepath


def plot_history(history,scoring: list=["auc"],validation_scoring: bool=True,log_scale: list=[False]):
    """Cette méthode permet de facilité l'affichage de l'historique d'entrainement de Keras.
    
arguments:
- history (type tensorflow.keras.callbacks.History) : Object retourner après l'entrainement d'un modèle Keras
- scoring (type list of str optionnel)(default ['auc']) : Specifie les scoring que l'on souhaite afficher (loss, metric)
- validation_scoring (type bool optionnel)(default True) : Specifie si l'on souhaite afficher le score de validation si il existe
- log_scale (type list of bool optionnel)(default [False]) : Permet d'afficher avec une echelle logarithmique ou non le scoring associée, par exemple: scoring=['accuracy','loss'] si on souhaite mettre l'echelle log sur 'loss' on fait log_scale=[False,True]
    """
    scoring = [x.lower() for x in scoring]
    size_epoch=1
    all_val = []
    plot_enable=True
    for score in scoring:
        if not score in history.history.keys():
            plot_enable=False
            break
    if plot_enable:
        fig, axs = plt.subplots(len(scoring),1,figsize=(20,len(scoring)*6))
        i=0
        for score in scoring:
            all_val = []
            for k in history.history.keys():
                if k in score:
                    axs[i].plot(history.history[k],label="train_set_"+k)
                    if validation_scoring:
                        axs[i].plot(history.history["val_"+k],label="validation_set_"+k)
                        all_val.extend(history.history["val_"+k])
                    all_val.extend(history.history[k])
                    
                    size_epoch = len(history.history[k])
            
            min_v = min(all_val)
            max_v = max(all_val)
            axs[i].set_ylim(min_v,max_v)
            axs[i].set_xticks([i for i in range(size_epoch)])
            axs[i].set_xlabel("Epochs")
            axs[i].legend()
            if len(log_scale) <= i:
                scale = False
            else:
                scale = log_scale[i]
                
            if scale:
                axs[i].set_yscale("log")
            i+=1
        plt.show()
    else:
        print("No scoring match on :", history.history.keys())
        
        
def get_all_prepared_data_no_augmented(path_X_list,path_Y_list,crop_x=512,crop_y=512,n_classes=8):
    """Cette méthode permet de recuperer une liste d'images formater pour réaliser par exemple des test du modèle.
    
arguments:
- path_X_list (type list of str) : list contenant les chemin d'accès aux input images.
- path_Y_list (type list of str) : list contenant les chemin d'accès aux masques.
- crop_x (type int optionnel)(default 512) : dimension de mise à l'echelle.
- crop_y (type int optionnel)(default 512) : dimension de mise à l'echelle.
- n_classes (type int optionnel)(default 8) : nombre de classes de notre segementation.

returns:
- X (type numpy.array): liste des images au format numpy
- Y (type numpy.array): liste des masques au format numpy
    """
    X=[]
    Y = []
    for path_X,path_Y in zip(path_X_list,path_Y_list):
        image= cv2.imread(path_X)
        mask = get_image_labelid(path_Y,one_hot_encoder=False)
        img_ = cv2.resize(image,(crop_x,crop_y))
        mask_= cv2.resize(mask,(crop_x,crop_y))
        Y.append(mask_)
        X.append(img_)
    X = np.array(X,dtype=np.uint8)
    Y = np.array(Y,dtype=np.uint8)
    if len(Y.shape) == 3:
        Y = np.expand_dims(Y, axis = 3)
    X = X /255. 
    train_masks_cat = to_categorical(Y, num_classes=n_classes)
    y = train_masks_cat.reshape((Y.shape[0], Y.shape[1], Y.shape[2], n_classes))
    return X,y
