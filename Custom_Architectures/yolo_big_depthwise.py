#--------------------#
# YOLO BIG DEPTHWISE #
#--------------------#

yolo = Sequential()

yolo.add(tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", strides = (2,2), input_shape=(448,448,3), use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.SeparableConv2D(64, (1, 1), padding="valid", use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", strides = (2,2), use_bias = False))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.Reshape((6272,), input_shape=(7,7,128)))

yolo.add(Dense(245, activation = 'sigmoid'))

yolo.add(tf.keras.layers.Reshape((7,7,5), input_shape=(245,)))

yolo.summary()
