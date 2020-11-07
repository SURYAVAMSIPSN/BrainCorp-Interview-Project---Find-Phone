Hello! My name is Surya Popuri from ASU, and this is the (possible) solution to the problem of phone detection given by Brain Corp

The return.py file accesses the images and labels, and converts them into numpy arrays, to be used by the train_phone_finder.py file.

The train_phone_finder.py trains the set of phone images using tensorflow and keras libraries, and the trained weight models
are saved as a .model file (You can run the .py file again if you want to retrain, but it will take a couple of mins)

The find_phone.py takes in an image input and tries to predict the co-ordinates of the phone. 

Comments are provided in the code for the evaluator to understand what is going on in the code. 



IMPORTANT NOTE!! Please follow the following instructions to run the .py files. 

Due to time constraints, both the .py files could not be programmed to be run on command prompt using a single command line argument. 
Some python console (Spyder, Jupyter, Pycharm, etc.) is needed to run. Also, all the sub-folders and associated files need to be within 
this folder as submitted in order to run without errors.

Additionally, there is a folder called find_phone_test_images which is supposed to contain the test images provided by the customer. 
Because the program was not designed for accepting single command line arguments, the evaluator needs to include the test images to be 
tested (the 8 test images) in that folder and then run the find_phone.py file in the console. For checking whether the code works or not, 
an image has already been included (30.jpg) in the folder for reference. In addition to this, the evaluator needs to add more images as 
required to test the performance of the trained model. 



