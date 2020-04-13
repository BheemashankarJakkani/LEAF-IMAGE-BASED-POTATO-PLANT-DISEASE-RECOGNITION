
import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
from keras.backend import clear_session

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
default_image_size = tuple((256, 256))
image_size = 0


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
        
# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        model = load_model(STATIC_FOLDER + '/' + 'trained_model.h5')
        imar = convert_image_to_array(full_name)
        npimagelist = np.array([imar], dtype=np.float16) / 225.0 
        PREDICTEDCLASSES2 = model.predict_classes(npimagelist) 
        predicted = model.predict(npimagelist)
        predicted_class = np.asscalar(np.argmax(predicted, axis=1))
        accuracy = round(predicted[0][predicted_class] * 100, 2)
        print(PREDICTEDCLASSES2)

        if PREDICTEDCLASSES2==2:
           label="Potato Healthy"
        elif PREDICTEDCLASSES2==1:
           label="Potato Late Blight"
        else:
           label="Potato Early Blight"

        clear_session()
    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy =accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
