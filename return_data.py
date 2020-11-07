import cv2
import numpy as np
import glob
import sys
import os

#NOTE!!! Please make sure all .py files, images, labels, etc. are in the same folder as this .py file. 

def return_data():
    # Stage 1: Accessing the labels. 
    path1 = r'find_phone\labels.txt'
    path2 = r'find_phone\*.jpg'
    # Since the labels are jumbled and the images are not, we try to sort the labels based on the jpg 
    # image name in order to sync with the images (Training data)
    list_of_labels = [line.rstrip('\n').split() for line in open(path1)] # Text to list.    
    # Next, we sort the list based on the JPG Image number. 
    for i in range(len(list_of_labels)):
        list_of_labels[i][0] = int(list_of_labels[i][0].replace('.jpg', ''))
    list_of_labels = sorted(list_of_labels) # sorts the labels by the jpg image number. 
    labels_cache = list_of_labels # might be useful later
    # Finally, we remove the image numbers from the sorted list, and what's left is a list of co-ordinates (float).
    for i in range(len(list_of_labels)):
        list_of_labels[i].pop(0)
        list_of_labels[i] = list(map(float, list_of_labels[i])) # Remove the image number and make co-ordinates float.
    
    # We now create a numpy array of corresponding labels. 
    y = np.array(list_of_labels) # Labels denoted usually by 'y'.
    # End of stage 1. 
    
    # Stage 2: Extracting images from the given folder. 
    X_data = []
    files = glob.glob (path2) # Access all the .jpg files. 
    img_size = 90
    for myFile in files:
        img = cv2.imread (myFile,0)
        #scale_percent = 70 # percent of original size
        #width = int(img.shape[1] * scale_percent / 100)
        #height = int(img.shape[0] * scale_percent / 100)
        #dim = (width, height)
        # resize image
        #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.resize(img, (img_size,img_size)) # make all images 80 x 80. 
        X_data.append(img)
    X = np.array(X_data) 
    X = X.reshape(-1, img_size,img_size,1)
    
    # Now we've prepared the data, ready to be exported to the main program. 
    return X,y