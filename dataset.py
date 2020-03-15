import numpy as np #array operations
import matplotlib.pyplot as plt #show img
import os #itarate through directiories and join paths
import cv2 #image operations
import random #can shuffle array 
import pickle #file package 

DATADIR = "C:/Users/JanMo/Downloads/Datasets/PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE = 75

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # transfer img to array
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) # resizes pictures for the same height and width
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

print(X[5])