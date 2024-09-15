from tf_keras.models import Sequential
from tf_keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.regularizers import l2
from tf_keras.callbacks import ReduceLROnPlateau
import pandas as pd

# Load and preprocess data
train_df = pd.read_csv("training/sign_mnist_train.csv")
test_df = pd.read_csv("training/sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,  # Enable horizontal flipping
    vertical_flip=False)

datagen.fit(x_train)

# Build the CNN model
model = Sequential()

# First block
model.add(Conv2D(100, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Second block
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Third block
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.3))  # Increased dropout
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Fourth block
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))  # Added L2 regularization
model.add(BatchNormalization())  # Added BatchNormalization after Dense
model.add(Dropout(0.4))  # Adjusted dropout
model.add(Dense(units=24, activation='softmax'))  # Output layer for 24 classes

# Compile the model with additional metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=256),  # Increased batch size
                    epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler])  # Added learning rate scheduler

# Save the model
model.save('smnist_improved.h5')
