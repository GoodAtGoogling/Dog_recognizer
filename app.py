# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:59:19 2023

@author: fondr
"""

from os.path import join
import os
from googletrans import Translator

image_dir = 'C:/Users/fondr/Desktop/Data Science Projects/Applications/Dog_recognizer/dog-breed-identification/train/'
img_paths = [join(image_dir, filename) for filename in 
                           ['000bec180eb18c7604dcecc8fe0dba07.jpg',
                            '00a338a92e4e7bf543340dc849230e75.jpg',
                            '00a366d4b4a9bbb6c8a63126697b7656.jpg',
                            '00ca18751837cd6a22813f8e221f7819.jpg',
                            '00a862390341c5be090dd72bd2bc19ef.jpg']]

import numpy as np
from keras.applications.resnet import preprocess_input
from keras.utils import load_img,img_to_array

translator = Translator()

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

from keras.applications import ResNet50



import tensorflow


my_model = ResNet50(weights='imagenet')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)
preds



from learntools.deep_learning.decode_predictions import decode_predictions
from IPython.display import Image, display


most_likely_labels = decode_predictions(preds, top=3, class_list_path='C:/Users/fondr/Desktop/Data Science Projects/Applications/Dog_recognizer/ResNet-50/imagenet_class_index.json')
breed_name = translator.translate(most_likely_labels[0][0][1], dest='fr').text
for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])