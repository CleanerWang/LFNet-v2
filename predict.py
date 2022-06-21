from yolo import YOLO
from PIL import Image
import os
yolo = YOLO()
image = Image.open("H:\\finished\\Cleaner_Wang\\LFNet_v2\\code\\code\\figs\\13.jpg")
r_image = yolo.detect_image(image)
r_image.show()
