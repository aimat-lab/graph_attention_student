"""
This example shows how the internal implementation of GnnExplainer can be used to generate explanations
for the RbMotifs dataset
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from imageio.v2 import imread
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes

import graph_attention_student.typing as tc
from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import array_normalize
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.data import load_eye_tracking_dataset_dict
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import NoLoss, ExplanationLoss
from graph_attention_student.training import mae
from graph_attention_student.models import gnnx_importances
from graph_attention_student.visualization import plot_node_importances, plot_edge_importances


# == DATASET PARAMETERS ==
DATASET_PATH = os.path.join(DATASETS_FOLDER, 'rb_dual_motifs')
METADATA_CONTAINS_INDICES = True
TRAIN_RATIO = 0.8
NUM_EXAMPLES = 100

# == MODEL PARAMETERS ==
# We only need the most simple of GNN's for the demonstration
class Model(ks.models.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.conv_layers = [GCN(units=k) for k in [5, 5, 5]]
        self.lay_pooling = PoolingNodes(pooling_method='sum')
        self.lay_dense = DenseEmbedding(units=1)

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs
        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])
        out = self.lay_pooling(x)
        out = self.lay_dense(out)
        return out

# == TRAINING PARAMETERS ==
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# == EVALUATION PARAMETERS ==
EVAL_BATCH_SIZE = 600

# == EXPERIMENT PARAMETERS ==
PATH = os.getcwd()
NAMESPACE = 'results/gnnx_example'
DEBUG = True
with Skippable(), (e := Experiment(base_path=PATH, namespace=NAMESPACE, glob=globals())):
    e.info('starting gnes example...')

    e.info('loading rb motifs dataset...')
    dataset_dict = load_eye_tracking_dataset_dict(DATASET_PATH, logger=e.logger)
    dataset_size = len(dataset_dict)
    dataset_index_dict = {}
    indices = []
    dataset: t.List[tc.GraphDict] = []
    for c, (name, data) in enumerate(dataset_dict.items()):
        metadata = data['metadata']
        g = metadata['graph']

        if METADATA_CONTAINS_INDICES:
            index = metadata['index']
        else:
            index = c

        g['node_importances'] = np.sum(np.array(g['multi_node_importances']), axis=-1, keepdims=True)
        g['edge_importances'] = np.sum(np.array(g['multi_edge_importances']), axis=-1, keepdims=True)

        g['edge_attributes'] = np.array(g['edge_attributes'], dtype=np.float)

        # With this here we want to support the possibility that the "value" annotation might already
        # be a multi target annotation (as is the case with classification datasets for example) but usually
        # this will be a single regression value and we will have to wrap that into a list first
        if isinstance(metadata['value'], list):
            g['graph_labels'] = np.array(metadata['value'])
        else:
            g['graph_labels'] = np.array([metadata['value']])

        indices.append(index)
        dataset_index_dict[index] = data

    dataset = [t[1]['metadata']['graph'] for t in sorted(dataset_index_dict.items(), key=lambda t: t[0])]
    e.info('determining train test split...')
    # At this point we need to determine the number of training samples exactly such that it is divisible
    # by the batch size! Because the implementation of GradCAM currently requires that
    num_train_samples = int(TRAIN_RATIO * dataset_size)
    if num_train_samples % BATCH_SIZE != 0:
        remaining = num_train_samples % BATCH_SIZE
        num_train_samples -= remaining

    train_indices = random.sample(indices, k=num_train_samples)
    test_indices = [index for index in indices if index not in train_indices]
    example_indices = random.sample(test_indices, k=NUM_EXAMPLES)
    x_train, y_train, x_test, y_test = process_graph_dataset(dataset, test_indices=test_indices)

    e.info('creating the model...')
    model: ks.models.Model = Model()
    model.compile(
        loss=[ks.losses.MeanSquaredError()],
        loss_weights=[1],
        metrics=[ks.metrics.MeanSquaredError()],
        optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
        run_eagerly=False
    )

    with tf.device('gpu:0'):
        # NOTE: validation during the training is explicitly disabled because the validation dataset would
        # have to be divisible by the batch size as well. As that would be too much of a hassle we rather
        # disable it and perform the assessment manually afterwards.
        hist = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            # validation_data=(x_test, y_test),
            # validation_freq=1,
            callbacks=LogProgressCallback(
                logger=e.logger,
                epoch_step=1,
                identifier=f'loss'
            ),
            verbose=0
        )
        e.info('finished model training')
        e.info(f'model with {model.count_params()} parameters')
        history = hist.history
        e['history'] = history

        e.info('calculating the gnnx explanations...')
        #out_test = y_test[0]
        #gnnx_importances(model, x=x_test, y=out_test, epochs=100)

        e.info('evaluating on the test set...')
        current = 0
        while current < len(test_indices):
            num_samples = min(len(test_indices) - current, EVAL_BATCH_SIZE)

            eval_indices = test_indices[current:current+num_samples]
            node_input = x_test[0][current:current+num_samples]
            edge_input = x_test[1][current:current+num_samples]
            edge_index_input = x_test[2][current:current+num_samples]

            out_true = y_test[0][current:current + num_samples]
            x_eval = (node_input, edge_input, edge_index_input)

            # ~ making predictions
            out_pred = model(x_eval)
            # ~ gnn explainer
            ni_pred, ei_pred = gnnx_importances(
                model,
                x_eval,
                out_pred,
                node_sparsity_factor=0.1,
                edge_sparsity_factor=0.1,
                learning_rate=0.01,
                epochs=250,
                logger=e.logger,
            )
            ni_pred = ni_pred.numpy()
            ei_pred = ei_pred.numpy()
            for c, index in enumerate(eval_indices):
                data = dataset_index_dict[index]
                g = data['metadata']['graph']
                e[f'out/pred/{index}'] = float(out_pred[c][0])
                e[f'out/true/{index}'] = float(out_true[c][0])
                e[f'ni/{index}'] = array_normalize(ni_pred[c])
                e[f'ei/{index}'] = array_normalize(ei_pred[c])

            current += num_samples
            e.info(f' * evaluated ({current}/{len(test_indices)})')

        e.info('calculating evaluation metrics...')
        errors = []
        ni_true = []
        ni_pred = []
        ei_true = []
        ei_pred = []
        for index in test_indices:
            data = dataset_index_dict[index]
            g = data['metadata']['graph']

            error = e[f'out/true/{index}'] - e[f'out/pred/{index}']
            errors.append(error)

            ni_pred += np.array(e[f'ni/{index}']).flatten().tolist()
            ni_true += np.array(g['node_importances']).flatten().tolist()

            ei_pred += np.array(e[f'ei/{index}']).flatten().tolist()
            ei_true += np.array(g['edge_importances']).flatten().tolist()

        mse = np.mean(np.square(errors))
        mae = np.mean(np.abs(errors))
        e['mse'] = mse
        e['mae'] = mae
        e.info(f' * mse: {mse}')

        r2 = r2_score(list(e['out/true'].values()), list(e['out/pred'].values()))
        e['r2'] = r2
        e.info(f' * r2: {r2}')

        ni_auc = roc_auc_score(ni_true, ni_pred)
        ei_auc = roc_auc_score(ei_true, ei_pred)
        e['ni_auc'] = ni_auc
        e['ei_auc'] = ei_auc
        e.info(f' * ni_auc: {ni_auc}')
        e.info(f' * ei_auc: {ei_auc}')

        e.info('drawing examples...')
        path = os.path.join(e.path, 'examples.pdf')
        num_channels = 1
        with PdfPages(path) as pdf:
            for c, index in enumerate(example_indices):
                ni = np.array(e[f'ni/{index}'])
                ei = np.array(e[f'ei/{index}'])

                data = dataset_index_dict[index]
                metadata = data['metadata']
                g = metadata['graph']
                g['node_coordinates'] = np.array(g['node_coordinates'])
                image = np.asarray(imread(data['image_path']))

                fig, rows = plt.subplots(ncols=2, nrows=num_channels, figsize=(16, 16),
                                         squeeze=False)

                for row_index in range(num_channels):
                    column_index = 0

                    # -- GROUND TRUTH --
                    ax_gt = rows[row_index][column_index]
                    ax_gt.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    node_importances = np.array(g['node_importances'])
                    plot_node_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(node_importances[:, row_index]),
                        node_importances=node_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    edge_importances = np.array(g['edge_importances'])
                    plot_edge_importances(
                        g=g,
                        ax=ax_gt,
                        vmax=np.max(edge_importances[:, row_index]),
                        edge_importances=edge_importances[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                    # -- EXPLANATIONS --
                    ax_mas = rows[row_index][column_index]
                    ax_mas.imshow(image, extent=(0, image.shape[0], 0, image.shape[1]))
                    plot_node_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ni),
                        node_importances=ni[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    plot_edge_importances(
                        g=g,
                        ax=ax_mas,
                        vmax=np.max(ei),
                        edge_importances=ei[:, row_index],
                        node_coordinates=g['node_coordinates']
                    )
                    column_index += 1

                pdf.savefig(fig)
                plt.close(fig)

