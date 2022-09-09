from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

K.clear_session()

#---------#
# YOLO v2 #
#------ --#
yolo =  Sequential()

yolo.add(Conv2D(filters=16, kernel_size= (7, 7), strides=(1, 1), input_shape =(448, 448, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=48, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=32, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=64, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=32, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=64, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=128, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=128, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=128, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=256, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=256, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(Conv2D(filters=256, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())

yolo.add(Flatten())
yolo.add(Dense(128))
yolo.add(Dense(256))
yolo.add(BatchNormalization())
yolo.add(Dense(245, activation='sigmoid'))

yolo.add(Reshape((7,7,5), input_shape=(245,)))
 
yolo.summary()
