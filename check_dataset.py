 
from os import listdir, remove
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFile, Image
# ImageFile.LOAD_TRUNCATED_IMAGES = True
mainpath = 'dataset/'
paths = ['test/', 'train/', 'val/']
folders = ['fractured/', 'not fractured/']

def remove_images():
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
                        print('removing:', filename)
                        remove(pathfolder+filename)
                        count +=1


def imshow(img, title = None):
    img = img      
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title != None:
        plt.title(title)
    plt.show()

def desnormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std[:, None, None] + mean[:, None, None]
    return img
  
def imshow2(images, labels, class_names, fontsize=10, ncol=7, nrow=5):
    num_images = len(images)
    assert num_images <= ncol * nrow, "Número de imagens excede a capacidade da grade."
    
    fig, axes = plt.subplots(nrow + 1, ncol, figsize=(15, 15))  # +1 para títulos
    axes = axes.flatten()  
    
    for col in range(ncol):
        if col < len(labels):
            axes[col].set_title(class_names[labels[col]], fontsize=fontsize)
        axes[col].axis('off')  # Desativa os eixos para os títulos
    
    for i, (img, label, ax) in enumerate(zip(images, labels, axes[ncol:])):
 
        npimg = img.numpy()
        # npimg = desnormalize(npimg)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.axis('off')
    
    for ax in axes[num_images+ncol:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    remove_images()