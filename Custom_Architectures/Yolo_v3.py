from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Flatten, LeakyReLU, AveragePooling2D, Conv2D, Reshape, Concatenate, Activation, Input, SeparableConv2D
tf.keras.backend.clear_session()

inp = Input((448,448,3))

x = Conv2D(8, (3, 3), padding="same", use_bias = False)(inp)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(16, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(8, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(16, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(32, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(16, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(32, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(16, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(32, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)


x = Conv2D(64, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(32, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(64, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(128, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(64, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(256, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(128, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(256, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(512, (3, 3), padding="same", strides=(2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(256, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(512, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(256, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(512, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Conv2D(256, (1, 1), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(256, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(256, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

out = Conv2D(256, (1, 1), padding="same", use_bias = False)(x)
out = BatchNormalization()(out)
out = LeakyReLU(alpha = 0.1)(out)
bbox = Conv2D(4, (1, 1), padding="same", name = "bbox",activation = "sigmoid", use_bias = False)(out)

out1 = Conv2D(256, (1, 1), padding="same", use_bias = False)(x)
out1 = BatchNormalization()(out1)
out1 = LeakyReLU(alpha = 0.1)(out1)
prob = Conv2D(1, (1, 1), padding="same", name = "prob",activation = "sigmoid", use_bias = False)(out1)

model = Model(inp,[bbox, prob])

model.summary()
