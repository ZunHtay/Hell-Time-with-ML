#Hello Dr Hoa, You'll need
#keras, tensorflow,pillow,pip,numpy,matplotlib please install pip first as its the package to install pakcages easily
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu')) #input shape was guessed // adjustable
                                                                                    # input shape is only needed in  the first layer as keras has
                                                                                    #an ability
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) # second layer filter size can be adjustable to 5x5, 7x7 , as i mntioned
                                                        # the layers can be added as much as needed to train neurons and for more parameters
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Adding a third Conv layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten()) #there is another layer which performs almost the same thing as flattening called drop out to avoid overfitting
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) # Dense is another layer for neurons, the layer has its own weight
                                                        # and act as a connection of neurons from previous layer and receive as an input
                                                        # 128 is the number of nodes // calculations neeeded
classifier.add(Dense(units = 1, activation = 'sigmoid')) #softmax layer can be used as well
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# adam is an optimizer/ algorithm of keras
# is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients
# Part 2 - preprocessing  the images
# image augmentation by flipping, blurring zooming the data
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Data/Train',
target_size = (128, 128),
batch_size = 16,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Data/Test',
target_size = (128, 128), # thi network will be formatting all the images into 128x128 since the input will be 128x128
batch_size = 16, # small batch size causes delay, on purpose of training the network with few samples (16) first and repeat the ptovess
class_mode = 'binary')
#fitting training set and test set to neural network to validate the accuracy per epoch.
classifier.fit_generator(training_set,
steps_per_epoch = (107/16), #divided by batch size
epochs = 25, #epochs is pass of all the training datas, in this case 25 passes
validation_data = test_set,
validation_steps = (57/16)) #divided by batch size
#making prediction and testing
from keras.preprocessing import image
test_image = image.load_img('Data/prediction/bro.jpg', target_size = (128, 128)) #loadinng test image and resizing it into 128x128
test_image = image.img_to_array(test_image) #turning image into an array i(w,h,3)
test_image = np.expand_dims(test_image, axis = 0) #This function expands the array by inserting a new axis at the
                                                  # specified position. inserting new array at position 0
result = classifier.predict(test_image)
training_set.class_indices
if result [0][0] == 0:
    prediction = 'camera'
else:
    prediction = 'cup'
