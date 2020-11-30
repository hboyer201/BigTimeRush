from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_convolution_data
from assignment import label_converter
from sklearn.metrics import mean_absolute_error
import time
import os
import tensorflow as tf
import numpy as np
import random


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        predicts yardage for plays.
        """
        super(Model, self).__init__()
        self.epsilon = 0.001


        self.batch_size = 30
        self.hidden_layer = 100
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.1)

        # TODO: Initialize all hyperparameters
        # TODO: Initialize all trainable parameters

        self.CNN_layer1 = tf.keras.layers.Conv2D(10, (2,2), padding="same")
        self.CNN_layer2 = tf.keras.layers.Conv2D(5, (2,2), padding="same")

        self.CNN_layer3 = tf.keras.layers.Conv2D(100, (2,2), padding="same")
        self.CNN_layer4 = tf.keras.layers.Conv2D(62, (2,2), padding="same")


        self.pooling_layer1 = tf.keras.layers.MaxPool2D((2,2))
        self.pooling_layer2 = tf.keras.layers.MaxPool2D((2,2))
        self.pooling_layer3 = tf.keras.layers.MaxPool2D((2,2))

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.relu2 = tf.keras.layers.LeakyReLU()
        self.relu3 = tf.keras.layers.LeakyReLU()
        self.relu4 = tf.keras.layers.LeakyReLU()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: plays, shape of (batch_size, 2, 11, 10)
        :return: logits - a matrix of shape (batch_size, num_classes)
        """

        lifting = tf.keras.layers.Dense(100)(inputs)

        inputs = tf.transpose(lifting, perm=[0,3,2,1])
        output = self.CNN_layer1(inputs)
        output = self.batch_norm1(output)
        output = self.relu1(output)
        output = self.pooling_layer1(output)

        output = self.CNN_layer2(output)
        output = self.batch_norm2(output)
        output = self.relu2(output)
        output = self.pooling_layer2(output)

        output = self.flatten_layer(output)

        output = self.dense1(output)
        output = self.relu4(output)
        output = self.dropout1(output)

        output = tf.reshape(output, (self.batch_size,-1))
        logits = self.dense2(output)
        return logits



    def loss(self, logits, labels):
        """
        Calculates the model MSE loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, num_classes)
        containing the result of multiple convolution and feed forward layers

        :param labels: during training, represent the correct number of yards rushed for each play.
        :return: the loss of the model as a Tensor
        """


        loss = tf.keras.losses.MSE(labels, logits)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels using MAE
        :param logits: a matrix of size (num_inputs, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes)
        :return: the accuracy of the model as a Tensor
        """
        if (len(logits) != len(labels)):
            print("ERROR: len legits != len labels")
        acc = mean_absolute_error(labels, logits)
        return acc

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    indices = tf.random.shuffle(list(range(len(train_inputs))))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    total_loss = 0
    num_batches = 0
    total_acc = 0


    for i in range(0, len(train_inputs), model.batch_size):
        batch_inputs = train_inputs[i:i + model.batch_size]
        batch_labels = train_labels[i:i + model.batch_size]
        batch_labels = label_converter(batch_labels)

        if (len(batch_inputs) < model.batch_size ):
            continue

        with tf.GradientTape() as tape:

            logits = model.call(batch_inputs)
            acc = model.accuracy(logits, np.int32(batch_labels))
            acc = int(acc)
            total_acc += int(acc)
            loss = model.loss(logits, batch_labels)
            total_loss += sum(loss)/len(loss)
        num_batches += 1
        gradients = tape.gradient(loss, model.trainable_variables)

        model.opt.apply_gradients(zip(gradients, model.trainable_variables))
    return total_acc/num_batches, total_loss/num_batches

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """
    correct = 0
    label_size = 0
    num_batches = 0
    total_loss= 0.0
    indices = tf.random.shuffle(list(range(len(test_inputs))))
    test_inputs = tf.gather(test_inputs, indices)
    test_labels = tf.gather(test_labels, indices)
    for i in range(0, len(test_inputs), model.batch_size):
        batch_inputs = test_inputs[i:i + model.batch_size]
        batch_labels = test_labels[i:i + model.batch_size]
        batch_labels = label_converter(batch_labels)

        if (len(batch_inputs) < model.batch_size):
            continue

        logits = model.call(batch_inputs)
        loss = model.loss(logits, batch_labels)
        total_loss += sum(loss)/len(loss)
        acc = model.accuracy(logits, batch_labels)
        acc = int(acc)

        num_batches += 1
        correct += acc
        label_size += 1

    return correct/num_batches, total_loss/num_batches




def main():
    '''
    Read in play data, initialize your model, and train and
    test your model for a number of epochs.
    :return: None
    '''
    data_file_path = "data/train.csv"

    train_inputs, train_labels, test_inputs, test_labels = get_convolution_data(data_file_path)

    model = Model()

    epochs = 10

    for i in range(0, epochs):
        acc,loss = train(model, train_inputs, train_labels)

    acc, loss = test(model, test_inputs, test_labels)
    print("total acc is", acc)


    return


if __name__ == '__main__':
    main()
