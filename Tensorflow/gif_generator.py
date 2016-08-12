import os
import time
from images2gif import writeGif
from PIL import Image
from os import listdir

DIR = "imgs/"
GIF_DIR = "gifs/"

def remove_previous_gifs():
    filelist = [f for f in os.listdir(GIF_DIR) if f.endswith(".gif")]
    for f in filelist:
        os.remove(GIF_DIR+f)

def remove_previous_imgs():
    filelist = [f for f in os.listdir(DIR)]
    for f in filelist:
        os.remove(DIR+f)

def create_gif():
    print("Getting list of files in", DIR)
    file_names = [f for f in listdir(DIR)]

    images = [Image.open(DIR + fn) for fn in file_names]

    size = (666,442)
    for im in images:
        im.thumbnail(size, Image.ANTIALIAS)

    print("Creating gif...")
    filename = GIF_DIR + time.strftime("%Y%m%d-%H%M%S") + ".gif"
    writeGif(filename, images, duration=0.2, subRectangles=False)

    print("Done")

# remove_previous_gifs()
create_gif()
remove_previous_imgs()