import numpy as np
from tensorflow.keras.models import load_model
import sys
import cv2

#NOTE!!! Please make sure all user-defined libraries, the folders containing train and 
# test images are in the same folder as this .py file.

# always a good idea to have an exception handler.
try:
    path = r'find_phone_test_images\aa'
    img_name = input('Image to be tested with (img_number.jpg): ')
    path = path.replace('aa',img_name)
    img = cv2.imread(path,0) 
    
    img_size = 90
    img = cv2.resize(img, (img_size,img_size))
    X_data = []
    X_data.append(img)
    X = np.array(X_data) 
    X = X.reshape(-1, img_size,img_size,1) # Reshaping is done for simplification and to adhere to the trained weights. 
    X = np.divide(X, 255)
    
    new_model_x = load_model('modelx.model')
    new_model_y = load_model('modely.model')
    
    print("%0.4f %0.4f"%(new_model_x.predict(X),new_model_y.predict(X))) # To print float digits side by side, no paranthesis.
except Exception as e:
    print(e)

