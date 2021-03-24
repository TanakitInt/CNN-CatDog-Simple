import os
from glob import glob
from os import path
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from pathlib import Path

# load weight
print("Loading trained weight...")

cnn = load_model('catdog-simple.h5')
print(cnn.summary())

# get file name and location
folder_p = Path('input/').rglob('*.png')
folder_j = Path('input/').rglob('*.jpg')
files_in = [x for x in folder_p] + [x for x in folder_j]

# remove old prediction file
try:
    os.remove("output/prediction.txt")
except:
    pass

# prediction loop
print("#### Prediction Results ####" + '\n')

f = open("output/prediction.txt", "a")
f.write("#### Prediction Results ####" + '\n')

count = 0

for i in files_in:
    
    #print(files_in[count])

    inputImage = files_in[count]

    # load image
    test_image = image.load_img(inputImage, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # predict image
    result = cnn.predict(test_image)

    if result[0][0] > 0.5:
        prediction = 'dog'

    else:
        prediction = 'cat'

    print(str(inputImage) + ": " + str(prediction) + '\n')

    # write into text file
    f = open("output/prediction.txt", "a")
    f.write(str(inputImage) + ": " + str(prediction) + '\n')

    count = count + 1

print("Write into file success!, Please check at \"output/\" folder.")

