import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
import os

class Control_Model():
    def __init__(self, input_len, output_len):

        self.out_classes = output_len

        # Building convolutional network
        network = input_data(shape=[None, input_len], name='input')
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.6)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.6)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.6)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.6)
        network = fully_connected(network, output_len, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy', name='target')

        self.comp_graph = network
        self.model = tflearn.DNN(self.comp_graph, tensorboard_verbose=0,
                                tensorboard_dir='tb_dir')
        self.has_weights = False

        if tf.train.latest_checkpoint('Model/') != None:
            self.model.load('Model/model.ckpt')
            self.has_weights = True


    def predict_move(self, observations, train=True):
        if self.has_weights:
            position_probabilities = self.model.predict(observations)
            position_probabilities = np.squeeze(position_probabilities)

            if not train:
                return np.argmax(position_probabilities)

            move = 0
            prev_prob = 0
            onehot_count = 0

            # pick a random number between 0 and 1
            uniform_num = np.random.uniform(0, 1)

            # check to see which 'window' the random number is in
            for probability in position_probabilities:
                if ((prev_prob <= uniform_num) and \
                    (uniform_num <= prev_prob + probability)):
                        move = onehot_count
                        break

                prev_prob += probability
                onehot_count += 1

            return move
        else:
            return None


    def train_game(self, features, labels):
        # Train the model
        labels = to_categorical(y=labels, nb_classes=self.out_classes)
        self.model.fit({'input': features}, {'target': labels}, n_epoch=5,
                  show_metric=False, batch_size=100, run_id='tensorboard_log',
                  shuffle=True)
        self.has_weights = True

        # remove all previous tensorboard files
        folder = 'tb_dir/tensorboard_log'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


    def save_model(self):
        self.model.save('Model/model.ckpt')















def ace_ventura():
    pass
