import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

no_tb_data = "TB_Chest_Radiography_Database/Normal"
tb_data = "TB_Chest_Radiography_Database/Tuberculosis"


X_yes = []
for image in tqdm(os.listdir(tb_data)):
    image_path = os.path.join(tb_data, image)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    X_yes.append(img)

x_yes = np.array(X_yes)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

aug_images = []
for image in tqdm(x_yes):  
    image = np.expand_dims(image, axis=0)  
    i = 0
    for batch in datagen.flow(image, batch_size=1):
        aug_images.append(batch[0])  
        i += 1
        if i >= 5:  
            break

TB_yes = []
for image in tqdm(aug_images):
    TB_yes.append([image, 1])

X_no = []
for image in tqdm(os.listdir(no_tb_data)):
    image_path = os.path.join(no_tb_data, image)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    X_no.append(img)
    
TB_no = []
for image in tqdm(X_no):
    TB_no.append([image, 0])

data = TB_yes + TB_no
random.shuffle(data)

X = []
y = []
for i, j in tqdm(data):
    X.append(i)
    y.append(j)
    
x = np.array(X)
y = np.array(y)


x_train = x[:5500]
y_train = y[:5500]
x_test = x[5500:7000]
y_test = y[5500:7000]

model = Sequential()

model.add(Conv2D(100,(3, 3), activation = "relu", input_shape = (224, 224, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(100,(3, 3), activation = "relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64,(3, 3), activation = "relu"))
#model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64,(3, 3), activation = "relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(.2))
#model.add(Dense(32, activation = "relu"))
model.add(Dropout(.3))
model.add(Dense(32, activation = "relu"))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split= .2, epochs = 5)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

threshold = 0.5
y_pred = (model.predict(x_test)>= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.show()

print(model.evaluate(x_test, y_test))