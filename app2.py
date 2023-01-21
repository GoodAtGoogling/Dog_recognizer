from flask import Flask, render_template, request, redirect, url_for
from os.path import join
import os
from keras.applications.resnet import preprocess_input
from keras.utils import load_img,img_to_array
from keras.applications import ResNet50
import tensorflow
from learntools.deep_learning.decode_predictions import decode_predictions

from os.path import join
import os
from keras.applications.resnet import preprocess_input
from keras.utils import load_img,img_to_array
from keras.applications import ResNet50
import tensorflow
from learntools.deep_learning.decode_predictions import decode_predictions
import numpy as np


app = Flask(__name__)

image_size = 224

# Load the pre-trained ResNet50 model
my_model = ResNet50(weights='imagenet')

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output

@app.route('/')
def home():
    return render_template('home.html')




@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if not os.path.exists('static/images'):
        os.mkdir('static/images')
    image.save(os.path.join('static/images', image.filename))
    img_paths = [os.path.join('static/images', image.filename)]
    # Preprocess the uploaded image
    test_data = read_and_prep_images(img_paths)
    # Use the model to make a prediction on the image
    prediction = my_model.predict(test_data)
    # Decode the prediction
    breed = decode_predictions(prediction,top=3,class_list_path='C:/Users/fondr/Desktop/Data Science Projects/Applications/Dog_recognizer/ResNet-50/imagenet_class_index.json')
    image_url = url_for('static', filename='images/' + image.filename)
    return render_template('result.html', breed=breed, image_url=image_url)







if __name__ == '__main__':
    app.run(debug=True)
