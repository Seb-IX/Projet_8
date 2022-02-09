import numpy as np
from flask import render_template, request,send_from_directory
from sklearn import datasets
from web import app
from web.utils import utils

from PIL import Image
from keras.preprocessing import image
import io
import os

#loading data
final_model = utils.load_model_azure()

NB_IMAGE = 20


@app.route("/",methods=["GET","POST"])
def index():
    response = {'success': False}
    if request.method == 'POST':
        if request.form.get('file'): # id is stored as name "file"
            id_img = request.form["file"]
            ## préprocessing input :
            data, size_x, size_y,path_img = utils.preprocess_data(id_img)
            y_pred = final_model.predict(data)
            mask = np.argmax(y_pred,axis=3)
            mask_color = utils.addColors(mask)
            mask_color = utils.resize_img(mask_color,size_x,size_y)
            # print("after resize:",mask_color.shape)
            # Formatage des predictions
            mask_color = Image.fromarray(mask_color)
            mask_color.save('./static/data/prediction/predicted_mask.png')
            mask_color.putalpha(120)
            img = Image.open(path_img)
            img.paste(mask_color, (0, 0), mask_color)
            response = utils.prepare_data(img)
            return render_template('prediction_image.html',sended=True,result=response,nb_image=NB_IMAGE)

    return render_template('prediction_image.html',sended=False,nb_image=NB_IMAGE)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')