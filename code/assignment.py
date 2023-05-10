import tensorflow as tf
import numpy as np
from preprocess import get_dataset
from types import SimpleNamespace
import os
import matplotlib.pyplot as plt
from matplotlib import colors

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        # make all of these different for each specific layer!
        # all conv layers should have 16 layers, except for resnet
        self.conv2D1 = tf.keras.layers.Conv2D(16, 3, padding='same')
        self.conv2D2 = tf.keras.layers.Conv2D(16, 3, padding='same')
        self.conv2D3 = tf.keras.layers.Conv2D(16, 3, padding='same')

        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.bnorm2 = tf.keras.layers.BatchNormalization()
        self.bnorm3 = tf.keras.layers.BatchNormalization()

        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        


    def call(self, x, training = False):
        # Define the forward pass of the residual block
        residual = x
        residual = self.conv2D1(residual)
        residual = self.bnorm1(residual, training=training)
        residual = self.dropout1(residual, training=training)
        
        body = x
        body = self.conv2D1(body)
        body = self.bnorm2(body, training=training)
        body = self.dropout2(body, training=training)
        body = self.conv2D3(body)
        body = self.bnorm3(body, training=training)
        x = tf.keras.layers.add([body, residual])
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, is_conv2d, num_filters=64):
        super(ResidualBlock, self).__init__()
        self.num_filters = num_filters

        self.conv2D1 = tf.keras.layers.Conv2D(num_filters, 3, padding='same')
        self.conv2D2 = tf.keras.layers.Conv2D(num_filters, 3, padding='same')
        self.conv2D3 = tf.keras.layers.Conv2D(num_filters, 3, padding='same')

        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.bnorm2 = tf.keras.layers.BatchNormalization()
        self.bnorm3 = tf.keras.layers.BatchNormalization()

        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.dropout4 = tf.keras.layers.Dropout(0.1)

        self.maxpool1 = tf.keras.layers.MaxPooling2D(2)
        self.maxpool2 = tf.keras.layers.MaxPooling2D(2)
        self.leaky = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.is_conv2d = is_conv2d
        

    def call(self, x, training = False):
        # Define the forward pass of the residual block
        residual = x
        residual = self.conv2D1(residual)
        if not self.is_conv2d:
            residual = self.maxpool1(residual, training=training)
        residual = self.bnorm1(residual, training=training)
        residual = self.dropout1(residual, training=training)
        
        body = x
        # leaky relu on body
        body = self.leaky(body)
        body = self.dropout2(body, training=training)
        if self.is_conv2d:
            body = self.conv2D2(body)
            body = self.bnorm2(body, training=training)
        else:
            body = self.maxpool2(body, training=training)
        body = self.leaky(body)
        body = self.dropout3(body, training=training)
        body = self.conv2D3(body)
        body = self.bnorm3(body, training=training)
        body = self.dropout4(body, training=training)
        x = tf.keras.layers.add([body, residual])
        return x


#########################################################################################

def get_model(input_shape):

    model = tf.keras.Sequential([
        # Down-sampling blocks
        Encoder(),
        # Residual blocks
        ResidualBlock(False, 32),
        ResidualBlock(False, 32),
        tf.keras.layers.UpSampling2D(2),
        ResidualBlock(True, 32),
        tf.keras.layers.UpSampling2D(2),
        ResidualBlock(True, 32),
        # delete this layer below but maybe keep sigmoid
        tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ])
    return model


#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    file_pattern = '../data/next_day_wildfire_spread_train*'
    import preprocess
    side_length = 32 #length of the side of the square you select (so, e.g. pick 64 if you don't want any random cropping)
    train_num_obs = 5000 #batch size
    train_data = preprocess.get_dataset(
      file_pattern,
      data_size=64,
      sample_size=side_length,
      batch_size=train_num_obs,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=True,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)
    inputs, labels = next(iter(train_data))
    # get only prev fire mask from inputs which is the last column



    val_num_obs = 1000 #batch size
    val_file_pattern = '../data/next_day_wildfire_spread_eval*'
    val_data = preprocess.get_dataset(
      val_file_pattern,
      data_size=64,
      sample_size=side_length,
      batch_size=val_num_obs,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=True,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)
    val_inputs, val_labels = next(iter(val_data))
    # get only prev fire mask from inputs which is the last column


    test_num_obs = 1000 #batch size
    test_file_pattern = '../data/next_day_wildfire_spread_test*'
    test_data = preprocess.get_dataset(
        test_file_pattern,
        data_size=64,
        sample_size=side_length,
        batch_size=test_num_obs,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False)
    test_inputs, test_labels = next(iter(test_data))
    # get only prev fire mask from inputs which is the last column

    train_batch_size = 128
    num_batches = train_num_obs // train_batch_size



    print(inputs.shape)
    print(labels.shape)
    n_rows = 5 
    # Number of data variables
    n_features = 12
    # Variables for controllong the color map for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.003, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    TITLES = [
    'Elevation',
    'Wind\ndirection',
    'Wind\nvelocity',
    'Min\ntemp',
    'Max\ntemp',
    'Humidity',
    'Precip',
    'Drought',
    'Vegetation',
    'Population\ndensity',
    'Energy\nrelease\ncomponent',
    'Previous\nfire\nmask',
    'Fire\nmask']

    

    # fit the sequential model
    model = get_model(inputs.shape[1:])
    # compile model with weighted cross entropy loss
    class_weights = tf.constant(20.0)

    # Create the loss function
    def weighted_cross_entropy_with_logits(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=class_weights)
    
    # add an AUC pr metric: MAYBE change to per pixel loss
    # l1, l2 regularization
    
    auc = tf.keras.metrics.AUC(curve='PR')
    model.build(inputs.shape)
    print(model.summary())
    print(labels.shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=weighted_cross_entropy_with_logits, metrics=[auc])
    model.fit(inputs, labels, epochs=100, batch_size=train_batch_size, validation_data=(val_inputs, val_labels), )
    # evaluate the model
    loss, acc = model.evaluate(test_inputs, test_labels, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    print('Test Loss: %.3f' % loss)
    labels = model.predict(test_inputs, batch_size=train_batch_size)
    print(np.shape(labels))
    inputs = test_inputs
    # make a prediction and evaluate it
    for i in range(n_rows):
      for j in range(n_features + 1):
        plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
        if i == 0:
          plt.title(TITLES[j], fontsize=13)
        if j < n_features - 1:
          plt.imshow(inputs[i, :, :, j], cmap='viridis')
        if j == n_features - 1:
          plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
        if j == n_features:
          plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM) 
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()