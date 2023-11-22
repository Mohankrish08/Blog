
We all know about the Convolution Neural Networks and how it works. In this blog we going to dig deep about the mathematical aspects about the convolution operation.

## How does the computer sees the images

Humans can easily differentiate colors and shapes to see the images, but coming with computer it can only understand the numbers, so we need to represented the images into numbers. 

![](Images/Pasted%20image%2020231122003842.png)


The above image is Gray scale image consists the values ranges from 0 to 255.
0 refers the most darkest region and 255 refers the brightest region in the image. By using this images are understand by the computer. 
When coming with RGB and BGR image, there be a combined stack of layers of Red, Green and Blue.  The each color channels the value ranges from 0 to 255, based on the values of the three color channels the images color also varies. 

## Introduction to Convolution Neural Nets

![](Images/Pasted%20image%2020231122005036.png)

The above image we identify the images as cat and dog by using the features of the animal like the whiskers, eyes and fur.
But how the Algorithm learns this is Cat or Dog??

### Convolution operation

It's a process of combing two functions to produce the third function. The output is known as feature map. We use a matrix, called a filter or kernel that is applied to the image. So, the image combined with kernel or filter to produce the output feature map.

`Image x Kernel = Feature map `


we already know about this, so I give a short intro about the operations.

**_Kernel_** - This is filter that extracts the features from the images.
**_Stride_** - Number of pixels that a filter moves over an input image.
**_Padding_** - Adding the extra pixel around the input image.
**_Pooling_** - Reduce the spatial dimension of the feature map.
**_Feature map_** - Output. 

![](Images/Pasted%20image%2020231122011529.png)
**_This the Formula_**

# Let's code

Take a Binary classification dataset contains with images of cats and dogs.

The structure of dataset is:

|-------train (training images)
|
| |------cats
| |------img 1.jpg
| |------img 2.jpg
| |------ ...
|
| |------dog
| |------img 1.jpg
| |------img 2.jpg
| 
|--------test (testing images)
|
| |-------cat
| |------img 1.jpg
| |------img 2.jpg
| |------ ...
|
| |------dog
| |------img 1.jpg
| |------img 2.jpg
| 

# Workflow of CNN model

![](Images/Pasted%20image%2020231122014436.png)

## Importing Libraries

```python
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib
import glob
```
## TensorFlow

```python
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
```
## Setup GPU

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
```
## Data

```python
dataset = 'Path of your dataset'
```
```python
for dirpath, dirnames, filenames in os.walk(dataset):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
```
## Converting images to Tensors

```python
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
```

### path for directory
	
```python 
train_dir = 'Path of the training data'
test_dir = 'Path of the testing data'
```

```python
train_data = train_gen.flow_from_directory(train_dir, batch_size=32, target_size=(224,224), class_mode='binary')
val_data = val_gen.flow_from_directory(test_dir, batch_size=32, target_size=(224,224), class_mode='binary')
```

# Custom model building

## Architecture

![](Images/Pasted%20image%2020231122161351.png)

```python
model = Sequential()

# first conv layer with 16 filerers
model.add(Conv2D(16, 3, 1, activation='relu', input_shape=(224,224, 3)))
model.add(MaxPool2D())

# second conv layer with 64 filters
model.add(Conv2D(64, 3, 1, activation='relu'))
model.add(MaxPool2D())

# third conv layers with 32 filters
model.add(Conv2D(32, 3, 1, activation='relu'))
model.add(MaxPool2D())

# flatten the layers
model.add(Flatten())

# Creating a fully connected layer
model.add(Dense(256, activation='relu'))

# adding the dropout layer
model.add(Dropout(0.5))

# final layer to produce the output
model.add(Dense(1, activation='sigmoid'))
```

**1. `First Conv2D (16, 3, 1)`**

It able to captures the low level features such as edges, simple textures, or basic shapes. Using 16 filters allows the network to learn the simple set of patterns.

**_2. `Maxpooling2D()`_**

Helps to reduces the spatial dimensions of the image. It focus on more abstract and high level features. It retains the important features of the previous layers. 

**3. `Second Conv2D(64, 3, 1)`**

It captures the complex features by combining information from the low-level features learned in the first layer. Using 64 filters allows the networks  to learn broader range of patterns.

**_4. `Maxpooling2D()`_**

Similar to the first pooling layer, it further reduces the spatial dimensions and retains important information.

**5. `Third Conv2D(32, 3, 1)`**

It refine the features learned in the previous layers. It has 16 filters, which may be fewer than the second layer because it's deeper in the network, and the features it captures are expected to be more abstract and specialized.

**_6. `Maxpooling2D()`_**

Helps to reduce the spatial dimension of the image. 

**7. `Dense()`**

It is a fully connected layer, it connects the neurons between the layers.

**8. `Dropout()`** 

It is used to randomly deactivating a portion of inputs units during training update. It helps to generalize the model by training the same weight again and again. 


The next step, we used see the summary of the model. For most of the people summary is a black box. We don't know about the internal working of the model architecture. But we are very familiarize with conceptual working of the model.

# Model Summary

![](Images/Pasted%20image%2020231122171142.png)

```python
model.summary()
```

![](Images/Pasted%20image%2020231122171233.png)

There are steps how does the image transforms in the every convolution. 

# Model Compile

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

# Model Fitting

```python
history = model.fit(train_data, validation_data=val_data, epochs=20)
```

# Plotting

Plot the accuracy and loss of the model.

### Accuracy

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
```

![](Images/Pasted%20image%2020231122183221.png)

### Loss

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```

![](Images/Pasted%20image%2020231122183248.png)

# Saving the model

Saving the model for further inference. 

```python 
model.save('Custom_cnn.h5')
```

# Loading the model for Inference

```python 
model = load_model('Custom_cnn.h5')
```

we write a function for inferencing the model. 

```python
def inference(path, model_path, target_size=(224,224)):
    model = load_model(model_path)
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)
    plt.imshow(img)
    predictions = model.predict(img_array)
    predictions_str = str(predictions)
    plt.title(predictions_str)
    plt.show()
    return predictions
```

The above function simply gets the input as path of the image and the trained model.
By using that, it used to convert the image into the numpy array, and predict the output for the image. 

```python
inference('cat 2.jpg', 'Custom_cnn.h5')
```

![](Images/Pasted%20image%2020231122190453.png)

This is the output the given image. 