from transformers import pipeline
from PIL import ImageDraw
from PIL import Image

import skimage
import numpy as np
import os
import cv2

from io import BytesIO

def tagImage(image_filename):
    image_numpy = skimage.io.imread(image_filename)
    img = Image.fromarray(np.uint8(image_numpy)).convert("RGB")
    

    predictions = detector(
        img,
        #candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner", "landmark", "karlskirche", "vienna", "karlsplatz", "mozarthaus"],
        candidate_labels=["burgtheater","austriacenter","oper","messe wien","rathaus","parlament",]
    )

    print(predictions)
    return img, predictions

def darwImage(img, predictions, i):
    draw = ImageDraw.Draw(img)

    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

    print(img)
    dir_path = os.path.dirname(os.path.realpath(result_folder))
    img.save(dir_path+ "/" + result_folder + "/" + str(i) +".jpg")
    
checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

sample_folder = "sample_from_wikipedia"
result_folder = "result_img"

#get the sample images
os.listdir(sample_folder)
i = 0
for filename in os.listdir(sample_folder):    
    if ".DS_Store" not in filename:
        print(filename)
        img, predictions = tagImage(sample_folder + "/" + filename)
        taggedImg = darwImage(img, predictions, i)
        i = i + 1
        


