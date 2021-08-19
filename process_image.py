from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import base64

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def preprocess_image(image_string):
    img = base64.b64decode(image_string)
    np_img_vector = np.fromstring(img, dtype=np.uint8) 
    
    image = cv2.imdecode(np_img_vector, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    return modified_image

def color_extraction_model(image_string):
    image = preprocess_image(image_string)

    num_colors = 4
    clusters = KMeans(n_clusters = num_colors)
    labels = clusters.fit_predict(image)

    counts = Counter(labels)

    center_colors = clusters.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]

    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return hex_colors
