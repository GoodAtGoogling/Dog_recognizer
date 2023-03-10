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

my_model = ResNet50(weights='imagenet')



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output



@app.route('/')
def home():
    return render_template('home.html')



from googletrans import Translator
translator = Translator()



@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if not os.path.exists('static/images'):
        os.mkdir('static/images')
    image.save(os.path.join('static/images', image.filename))
    img_paths = [os.path.join('static/images', image.filename)]

    test_data = read_and_prep_images(img_paths)

    prediction = my_model.predict(test_data)
    
 
    breed = decode_predictions(prediction,top=3,class_list_path='PATH/ResNet-50/imagenet_class_index.json') #Your pth to Resnet class index
    

    
 
    breed_name = breed[0][0][1] if breed[0][0][1] is not None else 'Unknown'
    
    breed_name_without_underscore = breed_name.replace("_", " ")
   
    try:
        breed_name = translator.translate(breed_name_without_underscore, dest='fr').text
    except:
        breed_name = breed[0][0][1]

    image_url = url_for('static', filename='images/' + image.filename)
    return render_template('result.html', breed_name=breed_name, breed=breed, image_url=image_url)





if __name__ == '__main__':
    app.run(debug=True)
