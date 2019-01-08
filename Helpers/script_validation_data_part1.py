# The purpose of this script is to make the val-data folder of the
# tiny-imagenet-200 the same structure as the training-data folder:
# Every label gets a folder that contains the data itself

# A problem that occurs with this script is the race condition:
# The folders have to be created first (part1.py) the files can only be
# copied when this is done (part2.py).

import re
import shutil
import os

filenames = []
label = []

with open("val_annotations.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)

for element in array:
    wordList = re.sub("[^\w]", " ",  element).split()
    #filename
    filenames.append(wordList[0] + "." + wordList[1])
    #label
    label.append(wordList[2])

i = 0
for element in filenames:
    source = "images/" + element
    destination = "output/" + label[i] + "/" + element
    shutil.copy(source, destination)
    i += 1
