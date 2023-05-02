"""
This example showcases how the final trained model can be saved permanently to the disk so that it can be
loaded and used at a later point without having to completely train the model again.

Since the MEGAN model is built on tensorflow/keras, the saving & loading process is rather easy with simply
calling the "save" and "load_model" functions. The model itself will be represented as a FOLDER on the
disk.
"""
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import load_visual_graph_dataset
from sklearn.metrics import r2_score

from graph_attention_student.models.megan import Megan
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.data import process_index_data_map
from graph_attention_student.training import LogProgressCallback, NoLoss

# ! NOTE: To run the example locally, you will have to download the corresponding visual graph dataset and
# insert the local path here.
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 50

__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    # ~ Loading the dataset
    e.log('loading the dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=e.VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
        metadata_contains_index=True,
    )

    dataset_indices, dataset = process_index_data_map(
        index_data_map,
        insert_empty_importances=True,
        importance_channels=2,
    )

    e.log('selecting the train-test split...')
    num_train = int(len(dataset_indices) * e.TRAIN_RATIO)
    train_indices = random.sample(dataset_indices, k=num_train)
    test_indices = list(set(dataset_indices).difference(set(train_indices)))

    e.log('creating tensors from dataset...')
    x_train, y_train, x_test, y_test = process_graph_dataset(
        dataset=dataset,
        test_indices=test_indices,
        train_indices=train_indices,
        use_importances=True,
    )

    # ~ Preparing the model
    e.log('creating the model...')
    model = Megan(
        units=[32, 32, 32],
        dropout_rate=0.15,
        final_units=[32, 16, 1],
        final_activation='linear',
        importance_channels=2,
        importance_factor=1.0,
        importance_multiplier=1.0,
        regression_reference=[-1.0],
        regression_weights=[[1.0, 2.0]],
        sparsity_factor=3.0,
        use_graph_attributes=False,
    )
    model.compile(
        loss=[
            ks.losses.MeanSquaredError(),
            NoLoss(),
            NoLoss(),
        ],
        loss_weights=[1, 0, 0],
        metrics=[ks.metrics.MeanSquaredError()],
        optimizer=ks.optimizers.Adam(learning_rate=0.001),
    )

    e.log('starting model training...')
    hist = model.fit(
        x_train, y_train,
        batch_size=e.BATCH_SIZE,
        epochs=e.EPOCHS,
        validation_freq=1,
        validation_data=(x_test, y_test),
        callbacks=LogProgressCallback(
            logger=e.logger,
            epoch_step=5,
            identifier='val_output_1_mean_squared_error'
        ),
        verbose=0,
    )
    history = hist.history
    e['history'] = history
    e.log(f'finished training of model with {model.count_params()} parameters')

    e.log(f'determining test set performance...')
    out_pred, ni_pred, ei_pred = [v.numpy() for v in model(x_test)]
    out_test, _, _ = y_test

    r2 = r2_score(out_test, out_pred)
    e['r2'] = r2

    e.log(f' * r2: {r2:.3f}')

    # ~ SAVING THE MODEL
    # This is how the model can be saved: Simply by calling the "save" method of the model and supplying
    # a folder path (which does already exist).
    # The tensorflow-keras model will be saved as a FOLDER consisting of several files which for example
    # contain the exact weights of the model. To load the model later on, this entire folder will be needed.
    e.log(f'Saving the model...')
    model_path = os.path.join(e.path, 'model')
    e['model_path'] = model_path
    model.save(model_path)


experiment.run_if_main()
