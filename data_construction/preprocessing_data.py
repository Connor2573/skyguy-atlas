from PIL import Image
import os

pathToRaw = r'./datasets/raw_images/maps/'
pathToNew = r'./datasets/dataset1/maps/'

def resizeImages(new_size):
    for filename in os.listdir(pathToRaw):
        f = os.path.join(pathToRaw, filename)
        if os.path.isfile(f) and f.find(".jpg") != -1:
            image = Image.open(f)
            resized = image.resize(new_size)
            resized.save(os.path.join(pathToNew, filename))

resizeImages((450, 300))