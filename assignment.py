import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl
import networkx as nx
import dgl.function as fn
torch.set_printoptions(threshold=sys.maxsize)
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error


message = fn.copy_src(src='features', out='m')
reduce = fn.sum(msg='m', out='features')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 3
        self.liftingLayer = nn.Linear(9, 200)
        self.gcn1 = GCN(200, 200)
        self.gcn2 = GCN(200, 200)
        self.gcn3 = GCN(200, 200)
        self.readout = nn.Linear(22*200, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)

    def forward(self, g):
        features = g.ndata.pop('features')
        x = torch.nn.functional.relu(self.liftingLayer(features))
        x = self.gcn1(g, x)
        x = self.gcn2(g, x)
        x = self.gcn3(g, x)
        x = x.reshape(self.batch_size, -1)
        x = self.readout(x)
        x = self.dropout(x)
        return x

    def accuracy_function(self, logits, labels):
        return mean_absolute_error(labels, logits)

class NodeModuleLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(NodeModuleLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, node):
        h = torch.relu(self.linear(node.data['features']))
        return {'features': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.mod = NodeModuleLayer(in_feats, out_feats)

    def forward(self, g, feature):
        g.ndata['features'] = feature
        g.update_all(message, reduce)
        g.apply_nodes(func=self.mod)
        return g.ndata.pop('features')


def build_graph(play):
    """
    Constructs a DGL graph out of a play from the train/test data.

    :param play: a play object (see play.py for more info)
    :return: A DGL Graph with the same number of nodes as atoms in the play, edges connecting them,
             and node features applied.
    """
    graph = dgl.DGLGraph()
    graph.add_nodes(len(play.nodes))
    tensor = torch.from_numpy(play.nodes)

    graph.ndata['features'] = tensor
    #      e.g if the edges of the play looked like [(1,2), (3,4), (5,6)] return
    #      (1,3,5) and (2,4,6).
    src = []
    dst = []
    for tuple in play.edges:
        src.append(tuple[0])
        dst.append(tuple[1])
    graph.add_edges(src, dst)
    graph.add_edges(dst, src)
    return graph


def train(model, train_data):
    """
    Trains your model given the training data.

    :param model: Model class representing your MPNN.
    :param train_data: A 1-D list of play objects, representing all the plays
    in the training set from get_data
    :return: nothing.
    """

    current_ball_carrier_index = 0
    rng_state = np.random.get_state()
    np.random.shuffle(train_data)
    loss = nn.MSELoss()
    for i in range(int(len(train_data) / model.batch_size)):
        offset = i * model.batch_size
        graphs = []
        labels = []
        for m in range(offset, offset + model.batch_size):
            G = build_graph(train_data[m])
            graphs.append(G)
            labels.append(train_data[m].label)

        labels_torch = torch.FloatTensor(np.array(label_scaler(labels)))
        batch = dgl.batch(graphs)
        labels_torch = labels_torch.reshape(model.batch_size, 1)
        x = Variable(model(batch), requires_grad=True)

        l = loss(x, labels_torch)
        model.optimizer.zero_grad()
        l.backward()
        model.optimizer.step()



def test(model, test_data):
    """
    Testing function for our model.

    Batch the plays in test_data, feed them into your model as described in train.
    After you have the logits: turn them back into numpy arrays, compare the MAE to the labels,
    and keep a running sum.

    :param model: Model class representing your MPNN.
    :param test_data: A 1-D list of play objects, representing all the plays in your
    testing set from get_data.
    :return: total MAE over the test set
    """
    tot_acc = 0
    num_batches = 0
    for i in range(int(len(test_data) / model.batch_size)):
        num_batches += 1
        offset = i * model.batch_size
        graphs = []
        labels = []
        for m in range(offset, offset + model.batch_size):
            graphs += [build_graph(test_data[m])]
            labels += [int(test_data[m].label)]
        labels = np.array(label_scaler(labels))
        batch = dgl.batch(graphs)
        logits = model(batch).detach().numpy()
        acc = model.accuracy_function(logits.reshape(model.batch_size), labels)
        tot_acc += acc
    return tot_acc / num_batches


def label_scaler(labels):
    newLabels = []
    for i in labels:
        label = i
        if label < -15:
            label = -15
        if label > 15:
            label = 15
        label += 15
        newLabels += [label]
    return newLabels


def main():
    # TODO: Return the training and testing data from get_data
    trainData, testData = preprocess.get_data('data/train.csv')
    print("finished preprocess")
    # TODO: Instantiate model
    model = Model()
    # TODO: Train and test for up to 15 epochs.
    for i in range(50):
        train(model, trainData)
        print("finished training epoch", i)
        acc = test(model, testData)
        print("Mean Absolute Error - epoch", i, "is", acc)


if __name__ == '__main__':
    main()
