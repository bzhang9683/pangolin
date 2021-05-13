#residual unit for ResNet-34 (ResNet with 34 layers)
#refer to the example code on the textbook

class ru_ResNET34(tf.keras.layers.Layers):
    def __init__(self, filters, strides = 1, activation = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, 3, strides = strides, padding = 'same', use_bias = False),
            tf.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides = 1, padding = 'same', use_bias = False),
            tf.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides = strides, padding = 'same', use_bias = False),
                tf.keras.layers.Conv2D.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
            return self.activation(Z+skip_Z)

#residual unit for ResNet-152 (ResNet with 152 layers)
#refer to the described architecture on the textbook

class ru_Bottleneck_ResNET152(tf.keras.layers.Layers):
    def __init__(self, filters, strides = 1, activation = 'relu', reducing_factor = 4, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters//reducing_factor, 1, padding = 'same', use_bias = False),
            tf.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters//reducing_factor, 3, strides = strides, padding = 'same', use_bias = False),
            tf.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 1, padding = 'same', use_bias = False),
            tf.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides = strides, padding = 'same', use_bias = False),
                tf.keras.layers.Conv2D.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
            return self.activation(Z+skip_Z)