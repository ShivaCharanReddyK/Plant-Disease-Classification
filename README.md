
---
# Smart AgriAid: Plant Disease Detection System
# Create Plant Disease Classification Model

This repository contains a Jupyter notebook for creating and saving a deep learning model to classify tomato diseases using TensorFlow and Keras. The model is trained on a dataset of tomato leaf images, which are categorized into five classes.

## Requirements

1. Python (version 3.7 or higher)
2. Jupyter Notebook
3. TensorFlow
4. Keras (part of TensorFlow 2.x)
5. Matplotlib
6. NumPy

## Installation

1. **Install Python**: If you haven't installed Python yet, download and install it from the official [Python website](https://www.python.org/downloads/).

2. **Install Jupyter Notebook**: Use pip to install Jupyter Notebook.
    ```bash
    pip install jupyter
    ```

3. **Install Required Libraries**: Use pip to install the required libraries.
    ```bash
    pip install tensorflow matplotlib numpy
    ```

## Example Dataset

The dataset should be organized in a directory with subdirectories for each class. For example:
```
tomato1/
    Tomato_Early_blight/
    Tomato_Late_blight/
    Tomato_Leaf_Mold/
    Tomato__Tomato_YellowLeaf__Curl_Virus/
    Tomato_healthy/
```

## Steps to Create and Save the Model

### 1. Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
```

### 2. Define Variables

```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
```

### 3. Import Image Data

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'tomato1',
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names
```

### 4. Split Data into Training, Validation, and Test Sets

```python
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
```

### 5. Create Dataset Pipeline

```python
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
```

### 6. Resize and Scale Image Data

```python
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1/255)
])
```

### 7. Augment Image Data

```python
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
```

### 8. Build the Model

```python
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 5

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)
model.summary()
```

### 9. Compile the Model

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

### 10. Train the Model

```python
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
```

### 11. Evaluate the Model

```python
model.evaluate(test_ds)
```

### 12. Plot Training and Validation Metrics

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

### 13. Save the Model

```python
model.save("tomato5.h5")
```

### 14. Load and Predict with the Model

To make predictions with the saved model:

```python
model = tf.keras.models.load_model('tomato5.h5')

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\nPredicted: {predicted_class}.\nConfidence: {confidence}%")
        plt.axis("off")
```

## Running the Flask Web Application

To run the model in a web application, use the Flask framework.

### Requirements

1. Flask
2. Flask-CORS
3. TensorFlow
4. Pillow
5. NumPy

### Installation

Use pip to install the required libraries.

```bash
pip install flask flask-cors tensorflow pillow numpy
```

### Running the Flask Application

To run the Flask application, execute the following command:

```bash
python app.py
```

## Conclusion

By following the steps outlined in this guide, you can create, train, evaluate, and save a model for classifying plant diseases. This guide also includes steps to load the saved model and make predictions on new data.

---
