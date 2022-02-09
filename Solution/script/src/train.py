import tensorflow as tf
# Si besoin de chargé un Model existant
from azureml.core import Datastore, Workspace
from azureml.core import Run

import model as mdl

from azureml.core import Model

run = Run.get_context()
ws = Workspace.from_config()

def create_checkpoint_path(root_dir,name):
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
    checkpoint_filepath = create_checkpoint_path(root_dir,name)
    monitor_get = "val_"+ metric.name
    
    model_checkpoint_callback_gen = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor_get,
        mode='max',
        save_best_only=True)
    return model_checkpoint_callback_gen, checkpoint_filepath

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

if __name__ == "__main__":
    # Train 
    # Récupération et création des generateur
    SIZE_IMG_X = 256
    SIZE_IMG_Y = 256
    INPUT_SHAPE = (SIZE_IMG_X,SIZE_IMG_Y,3)
    NB_CLASS = 8
    
    # Récupération du jeu de données
    datastore = Datastore.get(ws, 'workspaceblobstore')

    datastore.download("./")
    
    dir_y="./dataset/ouputs/"
    dir_x="./dataset/inputs/"
    y_train_path, y_val_path = get_all_file(dir_y,file_pattern="_labelIds.png")
    x_train_path, x_val_path = get_all_file(dir_x)
    
    
    data_train = mdl.GeneratorCitySpace(x_train_path,y_train_path,batch_size=1,crop_x=SIZE_IMG_X,crop_y=SIZE_IMG_Y)
    data_val = mdl.GeneratorCitySpace(x_val_path,y_val_path,batch_size=1,crop_x=SIZE_IMG_X,crop_y=SIZE_IMG_Y)
    
    # Création du modèle, de la métriques et du model checkpoint
    mean_iou_metric = mdl.Mean_IoU_custom()

    model_vgg16_unet = mdl.build_vgg16_unet(INPUT_SHAPE,n_classes=NB_CLASS)
    mdl.freeze_layer_vgg16(model_vgg16_unet)

    model_vgg16_unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou_metric])
    model_checkpoint_vgg16, file_weight_vgg16 = create_checkpoint_callback("./outputs/",
                                                                           "final_model",
                                                                           mean_iou_metric)
    _ = model_vgg16_unet.fit(data_train,  
                             verbose=1,
                             epochs=2,
#                              use_multiprocessing=True,
#                              workers=2,
                             validation_data=data_val,
                             callbacks=[model_checkpoint_vgg16])
    
    model_vgg16_unet.load_weights(file_weight_vgg16)
    model_vgg16_unet.save("./outputs/final_model")
    
#     model = run.register_model(model_name='final_model', 
#                            model_path='outputs/final_model',
#                            model_framework=Model.Framework.TENSORFLOW,
#                            model_framework_version='2.6',
#                            resource_configuration=ResourceConfiguration(cpu=2, memory_in_gb=4))
    # OR
    final_model = Model.register(model_path="./outputs/final_model",
                       model_name="final_model",
                       tags={'area': "computer_vision", 'type': "deep_learning"},
                       description="Modélisation final réaliser sur Azure, structure UNet en transfert learning \
                       avec poids VGG16 pour la partie encoder.",
                       workspace=ws)
    
