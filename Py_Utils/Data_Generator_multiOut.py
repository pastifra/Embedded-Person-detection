import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_list, bboxes_list, batch_size=25, dim=(448,448,3),
                 divisions=(7,14,28,56), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.path_list = path_list
        self.S = divisions
        self.bboxes_list = bboxes_list
        self.shuffle = shuffle
        self.on_epoch_end() #triggered at beginning and end of each epoch
        self.cell_size = [64,32,16,8]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
         # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_list))
        if self.shuffle == True: # For more robust data
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y_tot = []
        
        for k in range (0,len(self.S)):
            Y = np.empty((self.batch_size,self.S[k],self.S[k],5))
            batch_num = 0
            # Generate data
            for i in indexes:
                original_img = load_img(self.path_list[i])
                width, height = original_img.size
                # load the image with the required size and calculate scale factors
                image = load_img(self.path_list[i], target_size=(448, 448))
                scale_w = 448 / width 
                scale_h = 448 / height
                image = img_to_array(image)
                # scale pixel values to [0, 1]
                image = image.astype('float32')
                image /= 255.0
                y_img = np.zeros((self.S[k],self.S[k],5))
                for box in self.bboxes_list[i]:
                    xleft = int(box[0] * scale_w)
                    yleft = int(box[1] * scale_h)
                    b_width = int(box[2] * scale_w)
                    b_height = int(box[3] * scale_h)
                    ox = xleft + b_width/2
                    oy = yleft + b_height/2
                    # Calculate the coordinates of the cell in the grid that contains the center 
                    grid_col = trunc(ox/self.cell_size[k])
                    grid_row = trunc(oy/self.cell_size[k]) 
                    # Calculate the coordinates of the center of the bbox w.r.t the associated cell; (0,0) top left and (1,1) bottom right corners of the cell
                    ox_cell = (ox - (grid_col)*self.cell_size[k])/self.cell_size[k]
                    oy_cell = (oy - (grid_row)*self.cell_size[k])/self.cell_size[k]
                    # Calculate the width and height of the bbox in terms of cell size, a bbox of width 448/S(cell size) will have grid_width = 1
                    grid_width = b_width/self.cell_size[k]
                    grid_heigth = b_height/self.cell_size[k]
                    # Put the results into y; 1 represent the probability of the class
                    y = [1,ox_cell,oy_cell,grid_width,grid_heigth]
                    y_img[grid_row][grid_col] = y

                # Store sample
                X[batch_num,] = image

                # Store grid
                Y[batch_num,] = y_img

                batch_num += 1
                
            Y_tot.append(Y)
        return X, Y_tot
