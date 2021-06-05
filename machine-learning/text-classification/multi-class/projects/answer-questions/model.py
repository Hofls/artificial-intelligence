import tensorflow as tf


def train(text_generator, labels):
    unique_labels_count = len(set(labels))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(200, 8),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(unique_labels_count)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(text_generator, epochs=80, verbose=0)
    return {
        "model": model,
        "history": history
    }
