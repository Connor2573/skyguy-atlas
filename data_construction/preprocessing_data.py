from PIL import Image
import os

cat = 'maps'
pathToRaw = r'./datasets/raw_images/' + cat + '/'
pathToNew = r'./datasets/dataset1/' + cat + '/'

def resizeImages(new_size):
    for filename in os.listdir(pathToRaw):
        f = os.path.join(pathToRaw, filename)
        if os.path.isfile(f) and f.find(".jpg") != -1:
            image = Image.open(f)
            resized = image.resize(new_size)
            resized.save(os.path.join(pathToNew, filename))

resizeImages((768, 512))