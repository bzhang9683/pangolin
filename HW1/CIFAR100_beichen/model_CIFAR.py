import tensorflow as tf
class ru_ResNET34(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, 3, strides = strides, padding = 'same', use_bias = False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides = 1, padding = 'same', use_bias = False),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides = strides, padding = 'same', use_bias = False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z+skip_Z)

def model_def():
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64,7,strides = 2, input_shape = [32,32,3],
                                    padding = "same", use_bias = False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides = 2, padding = "same"))
    prev_filters = 64
    structure = [64]*1+[128]*2+[256]*1
    print(structure)
    for filters in structure:
        if filters == prev_filters:
            strides = 1 
        else:
            strides = 2
        model.add(ru_ResNET34(filters, strides = strides))
        prev_filters = filters
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='softmax'))
    #compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model




