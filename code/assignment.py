import tensorflow as tf
import numpy as np
from preprocess import get_dataset
from types import SimpleNamespace
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        # make all of these different for each specific layer!
        # all conv layers should have 16 layers, except for resnet
        self.conv2D1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.conv2D2 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.conv2D3 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')

        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.bnorm2 = tf.keras.layers.BatchNormalization()
        self.bnorm3 = tf.keras.layers.BatchNormalization()

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        


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

        self.conv2D1 = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same')
        self.conv2D2 = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same')
        self.conv2D3 = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same')

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


# class MyTrigram(tf.keras.Model):

#     def __init__(self, vocab_size, hidden_size=100, embed_size=64):
#         """
#         The Model class predicts the next words in a sequence.
#         : param vocab_size : The number of unique words in the data
#         : param hidden_size   : The size of your desired RNN
#         : param embed_size : The size of your latent embedding
#         """

#         super().__init__()

#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size

#         ## TODO: define your trainable variables and/or layers here. This should include an
#         ## embedding component, and any other variables/layers you require.
#         ## HINT: You may want to use tf.keras.layers.Embedding
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
#         self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
#         self.flatten = tf.keras.layers.Flatten()

        


#     def call(self, inputs):
#         """
#         You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
#         :param inputs: word ids of shape (batch_size, 2)
#         :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
#         """
#         inputs = self.embedding(inputs)
#         inputs = self.flatten(inputs)
#         inputs = self.dense1(inputs)
#         inputs = self.dense2(inputs)


#         return inputs

#     def generate_sentence(self, word1, word2, length, vocab):
#         """
#         Given initial 2 words, print out predicted sentence of targeted length.
#         (NOTE: you shouldn't need to make any changes to this function).
#         :param word1: string, first word
#         :param word2: string, second word
#         :param length: int, desired sentence length
#         :param vocab: dictionary, word to id mapping
#         """
#         reverse_vocab = {idx: word for word, idx in vocab.items()}
#         output_string = np.zeros((1, length), dtype=np.int32)
#         output_string[:, :2] = vocab[word1], vocab[word2]

#         for end in range(2, length):
#             start = end - 2
#             output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
#         text = [reverse_vocab[i] for i in list(output_string[0])]

#         print(" ".join(text))


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
    train_num_obs = 1000 #batch size
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
    num_batches = 50
    train_batch_size = train_num_obs // num_batches


    print(inputs.shape)
    print(labels.shape)

    # fit the sequential model
    model = get_model(inputs.shape[1:])
    # compile model with weighted cross entropy loss
    class_weights = tf.constant(3.0)

    # Create the loss function
    def weighted_cross_entropy_with_logits(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=class_weights)
    
    # add an AUC pr metric: MAYBE change to per pixel loss
    # l1, l2 regularization
    auc = tf.keras.metrics.AUC(curve='PR')
    model.build(inputs.shape)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=weighted_cross_entropy_with_logits, metrics=[auc])
    model.fit(inputs, labels, epochs=5, batch_size=train_batch_size, validation_data=(val_inputs, val_labels), )
    results = model.evaluate(test_inputs, test_labels, batch_size=train_batch_size)
    print("test loss, test acc:", results)

if __name__ == '__main__':
    main()