import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def hot(data_x):
  one_hot_x = [tf.keras.preprocessing.text.one_hot(d, 50) for d in data_x]
  padded_x = tf.keras.preprocessing.sequence.pad_sequences(one_hot_x, maxlen=4, padding = 'post')
  return padded_x

data_x = hot([ 'good',  'ok', 'nice', 'excellent'])
label_x = np.array([0,1,2,2])
print(data_x)

model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(50, 8, input_length=4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(12, activation='relu'),
  tf.keras.layers.Dense(3) # unique labels count
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(data_x, label_x, epochs=200, batch_size=2, verbose=0)
plt.plot(history.history['loss'])
print(model.predict(data_x))

