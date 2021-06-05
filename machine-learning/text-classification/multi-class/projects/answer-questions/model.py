import tensorflow as tf


def create(labels):
    unique_labels_count = len(set(labels))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(200, 128),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(unique_labels_count)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train(model, text_generator):
    history = model.fit(text_generator, epochs=80, verbose=0)
    return history
