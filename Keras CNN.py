# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
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
classifier.add(Dense(units = 1, activation = 'softmax')) #softmax layer can be used as well
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# adam is an optimizer/ algorithm of keras
# is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients