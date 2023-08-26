# Eduardo Arizala Dueñas

# Auto-setup when running on Google Colab
#if 'google.colab' in str(get_ipython()):
#!pip install openml
# General imports

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml as oml
import tensorflow.keras as keras

# Download FMINST data. Takes a while the first time.
mnist = oml.datasets.get_dataset(40996)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute, dataset_format='array');
X = X.reshape(70000, 28, 28)
fmnist_classes = {0:"T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 
                  6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# Take some random examples
from random import randint
fig, axes = plt.subplots(1, 5,  figsize=(10, 5))
for i in range(5):
    n = randint(0,70000)
    axes[i].imshow(X[n], cmap=plt.cm.gray_r)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_xlabel("{}".format(fmnist_classes[y[n]]))

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

print(X.shape) #(70000, 28, 28)

X = X.reshape((70000, 28 * 28))

X = X.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#X.shape, y.shape

print(X.shape) #(70000, 784)
print(y.shape) #(70000, 10)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")


# For Fashion MNIST, there exists a predefined stratified train-test split of 60000-10000. We therefore don't shuffle or stratify here.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, random_state=0)
#print(X_train)

Xf_train, x_val, yf_train, y_val = train_test_split(X_train, y_train, train_size=50000, shuffle=True, stratify=y_train, random_state=0)
#print(Xf_train)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#NO EJECUTAR

#tf.keras.layers.Dense(
    #units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    #bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    #activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    #**kwargs
#)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(28 * 28,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(28 * 28,)))
model.add(layers.Dense(512))
model.add(layers.ReLU(negative_slope=0.1)) # A leaky ReLU
model.add(layers.Dense(10, activation='softmax'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

## Add one more hidden layer for better performance
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model.summary()

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

# Shorthand
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
metrics=['accuracy'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

# Detailed
model.compile(loss=CategoricalCrossentropy(label_smoothing=0.01),
              optimizer=RMSprop(learning_rate=0.001, momentum=0.0),
              metrics=[Accuracy()])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#network.fit(X_train, y_train, epochs=3, batch_size=64)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def create_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=(28 * 28,)))
    model.add(layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model = create_model()
history = model.fit(Xf_train, yf_train, epochs=3, batch_size=64);

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model.to_json()

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

model = create_model()
history = model.fit(Xf_train, yf_train, epochs=3, batch_size=32, verbose=0, validation_data=(x_val, y_val))

history.history

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

predictions = model.predict(X_test)

# Visualize one of the predictions
sample_id = 0
print(predictions[sample_id])

np.set_printoptions(precision=7)
fig, axes = plt.subplots(1, 1, figsize=(2, 2))
axes.imshow(X_test[sample_id].reshape(28, 28), cmap=plt.cm.gray_r)
axes.set_xlabel("True label: {}".format(y_test[sample_id]))
axes.set_xticks([])
axes.set_yticks([])

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

from IPython.display import clear_output

# For plotting the learning curve in real time
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.max_acc = 0
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.max_acc = max(self.max_acc, logs.get('val_accuracy'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(8,3))
            plt.plot(N, self.losses, lw=2, c="b", linestyle="-", label = "train_loss")
            plt.plot(N, self.acc, lw=2, c="r", linestyle="-", label = "train_acc")
            plt.plot(N, self.val_losses, lw=2, c="b", linestyle=":", label = "val_loss")
            plt.plot(N, self.val_acc, lw=2, c="r", linestyle=":", label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}, Max Acc {:.4f}]".format(epoch, self.max_acc))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from sklearn.model_selection import train_test_split

plot_losses = TrainingPlot()
model = create_model()
history = model.fit(Xf_train, yf_train, epochs=25, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from tensorflow.keras import callbacks

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=3)
model = create_model()
history = model.fit(Xf_train, yf_train, epochs=25, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses, earlystop])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from tensorflow.keras import regularizers

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
model = create_model()
history = model.fit(Xf_train, yf_train, epochs=50, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses, earlystop])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
plot_losses = TrainingPlot()
history = model.fit(Xf_train, yf_train, epochs=50, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses, earlystop])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
plot_losses = TrainingPlot()
history = model.fit(Xf_train, yf_train, epochs=50, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses, earlystop]) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

network = models.Sequential()
network.add(layers.Dense(265, activation='relu', input_shape=(28 * 28,)))
network.add(layers.BatchNormalization())
network.add(layers.Dropout(0.3))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.Dropout(0.3))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.Dropout(0.3))
network.add(layers.Dense(10, activation='softmax'))
network.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
plot_losses = TrainingPlot()
history = network.fit(Xf_train, yf_train, epochs=50, batch_size=512, verbose=0, validation_data=(x_val, y_val), callbacks=[plot_losses, earlystop])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

#plt.show();

