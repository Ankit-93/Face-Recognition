import warnings
warnings.filterwarnings("ignore")
from facerec.architecture import * 
from facerec.mobilenet import *
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from keras.models import load_model
import pickle
import itertools
from scipy.spatial.distance import cosine

######pathsandvairables#########
face_data = './faceRecognize/facerec/data/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "./faceRecognize/facerec/weights/facenet_keras_weights.h5"
face_encoder.load_weights(path)
#face_encoder = build_mobilenetv2(3)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std
import matplotlib.pyplot as plt
import numpy as np

# Create a sample image (random data)
image = np.random.random((100, 100))

# Display the image using plt.imshow


for face_names in os.listdir(face_data):
    encodes = []
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)
        try:
            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            x = face_detector.detect_faces(img_RGB)
            x1, y1, width, height = x[0]['box']
            x1, y1 = abs(x1) , abs(y1)
            x2, y2 = x1+width , y1+height
            face = img_RGB[y1:y2 , x1:x2]
            
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

            if encodes:
                encode = np.sum(encodes, axis=0 )
                encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[face_names] = encode
        except:pass
            # image = plt.imread(image_path)
            # plt.imshow(image)  # cmap='gray' displays the image in grayscale
            # plt.axis('off')  # Turn off axis
            # plt.show()
    
path = './faceRecognize/facerec/encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)

indexes_range = range(len(encoding_dict.keys()))
two_digit_combinations = list(itertools.combinations(indexes_range, 2))
print("All 2-digit combinations for total classes:",len(encoding_dict.keys()))
for combination in two_digit_combinations:
    #print(combination[0],combination[1])
    print(f"Distance between {list(encoding_dict.keys())[combination[0]]} & {list(encoding_dict.keys())[combination[1]]} :{cosine(encoding_dict[list(encoding_dict.keys())[combination[0]]],encoding_dict[list(encoding_dict.keys())[combination[1]]])}")





