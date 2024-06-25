import os
from os import listdir
from PIL import Image
print('ok')

mainpath = 'dataset/'
paths = ['test/', 'train/', 'val/']
folders = ['fractured/', 'not fractured/']


for path in paths:
    for folder in folders:
        count = 0
        pathfolder = mainpath+path+folder
        print('checking', pathfolder)
        for filename in listdir(pathfolder):
            if filename.endswith('.jpg') or filename.endswith('.png') :
                try:
                    img = Image.open(pathfolder+filename).convert('RGB')  # open the image file
                    img.verify()  # verify that it is, in fact an image
                    img.close()
                except (IOError, SyntaxError) as e:
                    print(e, filename)
                    count +=1
        print("need to remove:", count)
