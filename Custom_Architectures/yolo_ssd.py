from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Flatten, LeakyReLU, AveragePooling2D, Conv2D, Reshape, Concatenate, Activation, Input, SeparableConv2D
tf.keras.backend.clear_session()

inp = Input((448,448,3))
x = Conv2D(16, (3, 3), padding="same", strides = (2,2), use_bias = False)(inp)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(16, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(16, (3, 3), padding="same", strides = (2,2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(32, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(32, (3, 3), padding="same", strides = (2,2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = SeparableConv2D(64, (3, 3), padding="same", strides = (2,2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)

x = SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
x = SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
out2 = Flatten()(x)
out2 = Dense(2048)(out2)
out2 = Dense(4096)(out2)
out2 = BatchNormalization()(out2)
out2 = Dense(3920, activation = 'sigmoid')(out2)
out2 = Reshape((28,28,5), input_shape=(3920,))(out2)

x = SeparableConv2D(128, (3, 3), padding="same",  strides = (2,2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
out1 = Flatten()(x)
out1 = Dense(512)(out1)
out1 = Dense(1024)(out1)
out1 = BatchNormalization()(out1)
out1 = Dense(980, activation = 'sigmoid')(out1)
out1 = Reshape((14,14,5), input_shape=(980,))(out1)

x = SeparableConv2D(256, (3, 3), padding="same", strides = (2,2), use_bias = False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.1)(x)
out = Flatten()(x)
out = Dense(128)(out)
out = Dense(256)(out)
out = BatchNormalization()(out)
out = Dense(245, activation = 'sigmoid')(out)
out = Reshape((7,7,5), input_shape=(245,))(out)

model = Model(inp,[out,out1,out2])

model.summary()
