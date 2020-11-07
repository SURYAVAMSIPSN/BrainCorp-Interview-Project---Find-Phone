import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
import numpy as np
from return_data import return_data # Where we already saved the data as numpy matrices (User-defined library) 
import pickle 

#NOTE!!! Please make sure all user-defined libraries, the folders containing train and 
# test images are in the same folder as this .py file.


# always a good idea to use exception handling. 
try:
    X, y = return_data() # Retrieve all data sets and labels. 
    # We now have a numpy array of training and testing data sets.
    
    X = np.divide(X, 255) # Normalization. 
    opt = Adadelta(learning_rate=1.0, rho=0.95) # Adadelta optimizer is used here, as for this problem, 
    # it gives the least error/cost. Parameters are set to default, as recommended by keras documentation. 
    
    
    # This is a regression problem, for a set of images. 
    # The labels are of multiple-output type, and no proper regression based algorithm has been officially developed. 
    
    # So, for this problem, the labels are split into two separate sets of labels, one for the x-coordinate
    # and one for the y-coordinate. 
    
    # For each type of label, training is separately done with two neural networks, each with their own 
    # activation functions, optimizers, weights and learning rates. 
    
    # Two different outputs are given, one is the x-coordinate and one is the y-coordinate. Both are printed together. 
    
    # First, we build the network for the x-cordinate label. 
    # The number of neurons and hidden layers and other numbers passed as parameters were used by trial and error.
    
    #layer 1
    modelx = Sequential() 
    modelx.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    modelx.add(Activation("relu"))
    modelx.add(MaxPooling2D(pool_size = (2,2)))
    
    #layer 2
    modelx.add(Conv2D(64, (3, 3)))
    modelx.add(Activation('relu'))
    modelx.add(MaxPooling2D(pool_size=(2, 2)))
    modelx.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    modelx.add(Dense(64))
    
    #layer 3
    modelx.add(Dense(1))
    modelx.add(Activation('sigmoid'))
    modelx.compile(loss = "mean_squared_error", optimizer = opt)
    
   
    # and now, a similar configuration for the y-coordinates. 
    modely = Sequential()
    modely.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    modely.add(Activation("relu"))
    modely.add(MaxPooling2D(pool_size = (2,2)))
    modely.add(Conv2D(64, (3, 3)))
    modely.add(Activation('relu'))
    modely.add(MaxPooling2D(pool_size=(2, 2)))
    modely.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    modely.add(Dense(64))
    modely.add(Dense(1))
    modely.add(Activation('sigmoid'))
    modely.compile(loss = "mean_squared_error", optimizer = opt)
    
    
    # The following two lines perform the training. Loss for each epoch is printed on the console window. 
    # First, modelx gets trained, then model y. 
    
    # Kindly wait until training is done!! For both the labels together, it shouldn't take more than 2 to 3  mins in 
    # worst case. 
    modelx.fit(X , y[:,0], batch_size = 10, epochs = 40, validation_split = 0.1) # All x co-ordinates
    modely.fit(X , y[:,1], batch_size = 10, epochs = 40, validation_split = 0.1) # All y co-ordinates
    
    
    
    # NOTE!!! The validation error doesn't change much due to the limited number of data available. 
    # However, the training error decreases significantly, and reaches a pretty minimum value. 
    # So, one could say that this algorithm isn't that bad! :) 
     
    
    
    # The following is to compare the predicted labels and desired labels, printed side by side, for the first 
    # 30 training examples (Can change the number in the range function). 
    
    # Uncomment this to run. 
    #for i in range(30):
    #    print(modelx.predict(X)[i], modely.predict(X)[i],' ', y[:,0][i], y[:,1][i])
    
    # The following saves the trained models, for later reuse.  
    # There might be a WARNING message, but it will get saved. (Hopefully)
    modelx.save('modelx.model')
    modely.save('modely.model')
    
     
    
except Exception as e:
    print("\nException!!!!")
    print(e)
    
### Shoutout to the Resources
# tensorflow, numpy, opencv and keras documentations
# sentdex - youtube channel
# sklearn documentation
# general python documentations. 
# github, stack-overflow for error queries. 
    
# Thank you to Brain-Corp for this wonderful opportunity!! :) Had fun coding. 