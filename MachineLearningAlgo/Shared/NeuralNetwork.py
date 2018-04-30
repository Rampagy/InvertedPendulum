import numpy as np
import tensorflow as tf
import tflearn
import os
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

class Control_Model():
    def __init__(self, input_len):
        # set file paths
        self.save_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Model')
        self.tb_name = 'tensorboard_log'

        # left or right
        self.out_classes = 2

        # Building convolutional network
        network = input_data(shape=[None, input_len], name='input')
        network = fully_connected(network, 100, activation='relu')
        network = dropout(network, 0.3)
        network = fully_connected(network, 100, activation='relu')
        network = dropout(network, 0.3)
        network = fully_connected(network, self.out_classes, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.00003,
                             loss='categorical_crossentropy', name='target')

        self.comp_graph = network
        self.model = tflearn.DNN(self.comp_graph, tensorboard_verbose=0,
                                tensorboard_dir=self.save_loc)
        self.has_weights = False

        if tf.train.latest_checkpoint(self.save_loc) != None:
            self.model.load(os.path.join(self.save_loc, 'model.ckpt'))
            self.has_weights = True


    def predict_move(self, observations, uniform=True):
        if self.has_weights:
            position_probabilities = self.model.predict(observations)
            position_probabilities = np.squeeze(position_probabilities)

            # if using uniform
            if uniform:
                position_probabilities = np.random.multinomial(1, position_probabilities, size=1)

            return np.argmax(position_probabilities)
        else:
            return np.random.randint(self.out_classes)


    def train_game(self, features, labels):
        # Train the model
        labels = to_categorical(y=labels.flatten(), nb_classes=self.out_classes)
        self.model.fit({'input': features}, {'target': labels}, n_epoch=2,
                  show_metric=False, batch_size=500, run_id=self.tb_name,
                  shuffle=True)
        self.has_weights = True

        # remove all previous tensorboard files
        folder = os.path.join(self.save_loc, self.tb_name)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


    def save_model(self):
        self.model.save(os.path.join(self.save_loc, 'model.ckpt'))
