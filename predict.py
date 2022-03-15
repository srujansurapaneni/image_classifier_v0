import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

batch_size = 32
class_names = {}

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image)
    prob_list = model.predict(image)
    
    
    classes = []
    probs = []
    
    rank = prob_list[0].argsort()[::-1]
    
    for i in range(top_k):
        
        index = rank[i] + 1
        cls = class_names[str(index)]
        
        probs.append(prob_list[0][index])
        classes.append(cls)
    
    return probs, classes

def process_image(img):
    img1 = tf.convert_to_tensor(img, dtype=tf.float32)
    img2 = tf.image.resize(img1, (224, 224))/255
    return img2.numpy()

def predict(path, model_loaded, top_k):
    if top_k < 1:
        top_k = 1
    img1 = Image.open(path)
    img2 = np.asarray(img1)
    img3 = process_image(img2)
    img4 = np.expand_dims(img3, axis=0)
    probe = model_loaded.predict(img4)
    classes = []
    probs = []
    rank = probe[0].argsort()[::-1]
    
    for i in range(top_k):
        
        index = rank[i] + 1
        cls = class_names[str(index)]
        
        probs.append(probe[0][index])
        classes.append(cls)
    
    return probs, classes

if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k
    if top_k is None: 
        top_k = 5

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)