import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle
from tensorflow.keras import backend as K

f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()


training_data = np.array(training_data[0].reshape(-1, 28, 28, 1)), training_data[1]
validation_data = np.array(validation_data[0].reshape(-1, 28, 28, 1)), validation_data[1]
test_data = np.array(test_data[0].reshape(-1, 28, 28, 1)), test_data[1]


initializer = tf.keras.initializers.HeNormal()

kernel = (3, 3)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel, activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(60, activation='relu', kernel_initializer=initializer))
model.add(layers.Dense(10, activation='relu', kernel_initializer=initializer))
model.add(layers.Softmax())




model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])



epoch = 10

start = time.time()
history = model.fit(training_data[0],
                    training_data[1],
                    epochs=epoch,
                    batch_size=100,
                    validation_data=(validation_data[0], validation_data[1]))
print(round((time.time()-start)*100)/100, "s")

plt.title("CNN "+str(kernel))
plt.plot(np.arange(0, epoch, 1), history.history['accuracy'], "o-", label='Dokładność')
plt.plot(np.arange(0, epoch, 1), history.history['val_accuracy'], "o-", label='Dokładność na zbiorze walidacyjnym')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.xticks(np.arange(1,epoch, 1))
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_data[0],  test_data[1], verbose=2)

print(test_acc)