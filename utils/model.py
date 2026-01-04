import tensorflow as tf
from tensorflow.keras import layers, models

# U-Net + LSTM Generator
def build_unet_lstm_generator(seq_len):
    assert seq_len == 375, "This version of the generator is hard-coded for input length = 375"
    inputs = tf.keras.Input(shape=(seq_len, 1))

    # Encoder
    c1 = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(inputs)    # 375 -> 188
    c2 = layers.Conv1D(128, 5, strides=2, padding='same', activation='relu')(c1)       # 188 -> 94
    c3 = layers.Conv1D(256, 5, strides=2, padding='same', activation='relu')(c2)       # 94 -> 47

    # LSTM Bottleneck
    b = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(c3)              # 47

    # Decoder
    u1 = layers.UpSampling1D(size=2)(b)  # 47 -> 94
    u1 = layers.Concatenate()([u1, c2])
    u1 = layers.Conv1D(128, 3, padding='same', activation='relu')(u1)

    u2 = layers.UpSampling1D(size=2)(u1)  # 94 -> 188
    u2 = layers.Concatenate()([u2, c1])
    u2 = layers.Conv1D(64, 3, padding='same', activation='relu')(u2)

    u3 = layers.UpSampling1D(size=2)(u2)  # 188 -> 376
    u3 = layers.Cropping1D(cropping=(0, 1))(u3)  # 376 -> 375 (match original length)
    u3 = layers.Conv1D(32, 3, padding='same', activation='relu')(u3)

    outputs = layers.Conv1D(1, 1, padding='same', activation='tanh')(u3)
    return tf.keras.Model(inputs, outputs, name='unet_lstm_generator_static')


# Enhanced Discriminator with LSTM
def build_enhanced_discriminator(seq_len):
    inputs = tf.keras.Input(shape=(seq_len, 1))
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name='enhanced_discriminator')

# Loss functions
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def gen_loss(fake_output, real_ecg, fake_ecg):
    adv_loss = loss_fn(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_ecg - fake_ecg))
    return adv_loss + 100 * l1_loss

def disc_loss(real_output, fake_output):
    return loss_fn(tf.ones_like(real_output), real_output) + \
           loss_fn(tf.zeros_like(fake_output), fake_output)
gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# ECG Classification Model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
        
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model