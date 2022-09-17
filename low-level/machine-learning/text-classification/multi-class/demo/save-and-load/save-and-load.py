import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob, os

def to_numbers(text):
  numbers = []
  for symbol in list(text):
    numbers.append(ord(symbol))
  return np.asarray(numbers).astype('float32')

data_x = np.array([to_numbers('good'), to_numbers('ok'), to_numbers('nice'), to_numbers('excellent')])
data_x = tf.keras.preprocessing.sequence.pad_sequences(data_x, padding='post')
label_x = np.array([0,1,2,2])
print(data_x)

model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(200, 128),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(3) # unique labels count
 ])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(data_x, label_x, epochs=200, verbose=0)
plt.plot(history.history['loss'])
print("Normal model predictions:")
print(model.predict(data_x))

# Save
current_directory = os.path.dirname(os.path.realpath(__file__))
path_to_model = os.path.join(current_directory, "exported-model")
model.save(path_to_model, save_format='tf')

# Load
print("Loaded model predictions:")
loaded_model = tf.keras.models.load_model(path_to_model)
print(loaded_model.predict(data_x))