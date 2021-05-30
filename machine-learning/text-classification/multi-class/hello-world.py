import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def hot(data_x):
  one_hot_x = [tf.keras.preprocessing.text.one_hot(d, 50) for d in data_x]
  print(one_hot_x)
  padded_x = tf.keras.preprocessing.sequence.pad_sequences(one_hot_x, maxlen=4, padding = 'post')
  print(padded_x)
  return padded_x

def predict(model, word):
    one_hot_word = [tf.keras.preprocessing.text.one_hot(word, 50)]
    pad_word = tf.keras.preprocessing.sequence.pad_sequences(one_hot_word, maxlen=4,  padding='post')
    print(pad_word)
    print(model.predict(pad_word))


data_x = hot([ 'good',  'well', 'nice', 'excellent'])

label_x = np.array([0,0,1,1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50, 8, input_length=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(2)
 ])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(data_x, label_x, epochs=200, batch_size=2, verbose=0)

plt.plot(history.history['loss'])

predict(model, 'excellent')

