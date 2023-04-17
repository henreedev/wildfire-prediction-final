import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param hidden_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        ## TODO: define your trainable variables and/or layers here. This should include an
        ## embedding component, and any other variables/layers you require.
        ## HINT: You may want to use tf.keras.layers.Embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.flatten = tf.keras.layers.Flatten()

        


    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        inputs = self.embedding(inputs)
        inputs = self.flatten(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)


        return inputs

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.
        (NOTE: you shouldn't need to make any changes to this function).
        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN


    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    def perplexity(y_true, y_pred):
        return tf.exp(tf.reduce_mean(loss_metric(y_true, y_pred)))
    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=loss_metric, 
        metrics=[perplexity],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )


#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    data_path = "../data"
    import preprocess
    train_words_tokenized, test_words_tokenized, word_to_token_dict = preprocess.get_data(f"{data_path}/train.txt", f"{data_path}/test.txt")

    vocab = word_to_token_dict

    def process_trigram_data(data):
        X = np.array(data[:-1])
        Y = np.array(data[2:])
        X = np.column_stack((X[:-1], X[1:]))
        return X, Y

    X0, Y0 = process_trigram_data(train_words_tokenized)
    X1, Y1 = process_trigram_data(test_words_tokenized)  




    # TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    # TODO: Implement get_text_model to return the model that you want to use. 
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab: print(f"{word1} not in vocabulary")
        if word2 not in vocab: print(f"{word2} not in vocabulary")
        else: args.model.generate_sentence(word1, word2, 20, vocab)

if __name__ == '__main__':
    main()