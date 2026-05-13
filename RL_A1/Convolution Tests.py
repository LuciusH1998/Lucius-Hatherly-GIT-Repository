# Importing relevant packages and libraries 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import requests
#from io import BytesIO

# Importing relevant functions from utils.py 
from utils import discrete_2d_convolution

# Loading Image
#url = "https://es.motor1.com/news/731567/fiat-multipla-matriculado-nuevo-2024/"
# Direct image extraction workaround (Motor1 page contains embedded images)
# We manually load one of the image links from the page:
# image_url = "https://es.motor1.com/photos/701355/fiat-multipla-monovolumen-antiguo-feo/#4437967_fiat-multipla-monovolumen-antiguo-feo"

# response = requests.get(image_url)
# Converting to Grayscale
image = Image.open("assignment1_release/Car_Image.jpg").convert("L")  
image = np.array(image)

# Question 2 Applying Kernel 
kernel_blur = np.ones((7,7))/49
image_blurred = discrete_2d_convolution(image, kernel_blur)

# Question 3 Edge Detection Kernel

# Defining the sobel vertical edge Detector
vertical_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Defining the horizontal edge Detector
horizontal_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Defining the vertical and horizontal edges 
vertical_edges = discrete_2d_convolution(image, vertical_kernel)
horizontal_edges = discrete_2d_convolution(image, horizontal_kernel)

# Displaying Results 
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image Plot")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Blurred Image (7x7 Average (1/49))")
plt.imshow(image_blurred, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Vertical Edges Graph")
plt.imshow(vertical_edges, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Horizontal Edges Graph")
plt.imshow(horizontal_edges, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()