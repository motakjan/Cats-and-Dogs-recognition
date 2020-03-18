import cv2
import tensorflow as tf
import os #itarate through directiories and join paths
import matplotlib.pyplot as plt #show img
    
CATEGORIES = ["Dog","Cat"]
IMAGEDIR = "C:/Users/JanMo/Desktop/Skola/Neural networks/catsAndDogs/images"

def prepare(filepath):
    IMG_SIZE = 75
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predictData():
    for img in os.listdir(IMAGEDIR):
        prediction = model.predict([prepare(os.path.join(IMAGEDIR,img))]) # always predict a list
        print("Prediction: " + CATEGORIES[int(prediction[0][0])] + " " + "Reality: " + str(img))
        #img_array = cv2.imread(os.path.join(IMAGEDIR,img), cv2.IMREAD_GRAYSCALE)
        #new_array = cv2.resize(img_array,(75,75))
        #plt.imshow(new_array, cmap="gray")
        #plt.show()
     
model = tf.keras.models.load_model("64x3-CNN-new.model")

predictData()



