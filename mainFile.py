import cv2
import tensorflow as tf
import os #itarate through directiories and join paths
    
CATEGORIES = ["Dog","Cat"]
IMAGEDIR = "C:/Users/JanMo/Desktop/Skola/Neural networks/catsAndDogs/images"

def prepare(filepath):
    IMG_SIZE = 75
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predictData():
    for img in os.listdir(IMAGEDIR):
        prediction = model.predict([prepare(str(img))]) # always predict a list
        print("Prediction")
        print(CATEGORIES[int(prediction[0][0])])
        print("Reality")
        print(img)
     
model = tf.keras.models.load_model("64x3-CNN.model")

predictData()



