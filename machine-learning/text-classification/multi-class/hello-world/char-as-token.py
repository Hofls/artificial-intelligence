import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def toNumbers(text):
  numbers = []
  for symbol in list(text):
    numbers.append(ord(symbol))
  return np.asarray(numbers).astype('float32')

def predict(text):
  converted = np.array([toNumbers(text)])
  print(model.predict(converted))

data_x = np.array([toNumbers('good'), toNumbers('ok'), toNumbers('nice'), toNumbers('excellent')])
data_x = tf.keras.preprocessing.sequence.pad_sequences(data_x, padding='post')
print(data_x)
label_x = np.array([0,0,1,1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(1000, 8),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(2)
 ])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(data_x, label_x, epochs=200, batch_size=2, verbose=0)

plt.plot(history.history['loss'])

predict('ok')


