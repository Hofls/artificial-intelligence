import tensorflow as tf
import matplotlib.pyplot as plt
import augment
import input

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(200, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)  # unique labels count
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
rawData = input.get_training_data()
textGenerator = augment.TextAugmentation(rawData["questions"], rawData["labels"])
history = model.fit(textGenerator, epochs=80, verbose=0)
plt.plot(history.history['loss'])
plt.show()

print(rawData["questions"])
print(augment.augment_each(rawData["questions"]))
augmentedData = textGenerator.__getitem__(0)
print(model.predict(augmentedData))


# get-training-data
# Commented out: import-model
# create-model
# train-model
# print-results
# Commented out: export-model
