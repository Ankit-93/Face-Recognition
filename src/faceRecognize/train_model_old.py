import face_recognition
import pickle
import cv2
import os
from imutils import paths
from tqdm import tqdm

def load_image(imagePath):
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    return rgb, boxes, encodings

def process_image(imagePath, knownEncodings, knownNames, pbar):
    name = imagePath.split(os.path.sep)[-2]
    rgb, boxes, encodings = load_image(imagePath)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

    pbar.update(1)

def process_images_in_chunks(image_folder, output_file, chunk_size=50):
    knownEncodings = []
    knownNames = []
    imagePaths = list(paths.list_images(image_folder))
    total_images = len(imagePaths)

    with tqdm(total=total_images) as pbar:
        for i in range(0, total_images, chunk_size):
            batch_paths = imagePaths[i:i + chunk_size]
            for imagePath in batch_paths:
                process_image(imagePath, knownEncodings, knownNames, pbar)

    serialize_encodings(output_file, knownEncodings, knownNames)

def serialize_encodings(output_file, knownEncodings, knownNames):
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(data))

if __name__ == "__main__":
    image_folder = "dataset"
    output_file = "encodings.pickle"
    process_images_in_chunks(image_folder, output_file)
