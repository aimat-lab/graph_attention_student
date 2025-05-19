import os
import gc
from typing import List, Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint
from scipy.special import softmax
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import auc
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.utils import np_array
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.megan import MeganEnsemble
from graph_attention_student.metrics import rll_score
from graph_attention_student.metrics import threshold_error_reduction
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import plot_threshold_error_reductions
from visual_graph_datasets.visualization.importances import create_importances_pdf
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from sklearn.metrics import accuracy_score, f1_score

# == EXPERIMENT PARAMETERS ==

# :param IDENTIFIER:
#       This string is used to identify the experiment in the experiment log files and the output
#       folder structure. It can be used to identify grouping of experiments.
IDENTIFIER: str = 'default'
# :param SEED:
#       This integer number is used to seed the random number generators of the experiment. This
#       is important for reproducibility of the results.
SEED: int = 40

# == DATASET PARAMETERS ==

# :param BASE_EXPERIMENT:
#       The name of the base experiment which should be used to construct an individual ensemble 
#       member. This should be the name of one of the existing experiment modules in the same
#       experiment folder.
BASE_EXPERIMENT: str = 'vgd_torch__megan__mutagenicity.py'
# :param DATASET_TYPE:
#       Either 'regression' or 'classification'. This determines the type of the dataset and how
#       the model is trained and evaluated. This will automatically be set by the base experiment
#       module.
DATASET_TYPE: Optional[str] = None
# :param CHANNEL_INFOS:
#       This dictionary structure can be used to define the human readable names for the various
#       channels that are part of the dataset. The keys of this dict have to be integer indices of
#       the channels in the order as they appear in the dataset.
#       Will be set by the base experiment module.
CHANNEL_INFOS: Optional[dict] = None

# == MODEL PARAMETERS ==

# :param IMPORTANCE_OFFSET:
#       This parameter controls the sparsity of the explanation masks even more so than the sparsity factor.
#       It basically provides the upper limit of how many nodes/edges need to be activated for a channel to 
#       be considered as active. The higher this value, the less sparse the explanations will be.
#       Typical values range from 0.2 - 2.0 but also depend on the graph size and the specific problem at 
#       hand. This is a parameter with which one has to experiment until a good trade-off is found!
IMPORTANCE_OFFSET: Optional[float] = 1.5
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the 
#       explanation co-training, this determines the margin for the thresholding. Instead of using the regression
#       reference as a hard threshold, values have to be at least this margin value lower/higher than the 
#       regression reference to be considered a class sample.
REGRESSION_MARGIN: Optional[float] = -0.1

# == ENSEMBLE PARAMETERS ==

# :param NUM_MODELS:
#       This integer number determines how many models to train in the ensemble.
NUM_MODELS: int = 5
# :param EPOCHS:
#       This integer number determines how many epochs to train each individual model in the ensemble.
EPOCHS: int = 150
# :param LEARNING_RATE:
#       This float number determines the learning rate for the training of the individual models in
#       the ensemble.
LEARNING_RATE: float = 1e-5
# :param BATCH_SIZE:
#       This integer number determines the batch size for the training of the individual models in
#       the ensemble.
BATCH_SIZE: int = 32
# :param NUM_TRAIN:
#       This float number determines the fraction of the dataset that should be used for training
#       the individual models in the ensemble.
NUM_TRAIN: Union[int, float] = 0.95
# :param NUM_VAL:
#       This float number determines the fraction of the dataset that should be used for validation
#       the individual models in the ensemble.
NUM_VAL: Union[int, float] = 0.1
# :param NUM_TEST:
#       This float number determines the fraction of the dataset that should be used for testing
#       the individual models in the ensemble.
NUM_TEST: Union[int, float] = 0.1

# == VISUALIZATION PARAMETERS ==

# :param FIG_SIZE:
#       This float number determines the size of the figures that are generated during the evaluation
#       of the ensemble model.
FIG_SIZE: float = 6.0

__DEBUG__: str = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('calibrate_model', default=False, replace=True)
def calibrate_model(e: Experiment,
                    model: MeganEnsemble,
                    index_data_map: dict,
                    indices: List[int],
                    **kwargs,
                    ) -> None:
    
    graphs: List[dict] = [index_data_map[i]['metadata']['graph'] for i in indices]
    values_true = [graph['graph_labels'] for graph in graphs]
    
    model.stack_predictions(graphs, values_true)
    
    results: List[dict] = model.forward_graphs(graphs)
    values_pred = [result['graph_output'] for result in results]
    errors = np.array([np.abs(true - pred) for true, pred in zip(values_true, values_pred)])
    
    # model.calibrate_uncertainty(graphs, errors)


@experiment.hook('evaluate_ensemble', default=False, replace=True)
def evaluate_ensemble(e: Experiment,
                      model: MeganEnsemble,
                      index_data_map: dict,
                      indices: List[int],
                      key: str = 'default',
                      **kwargs,
                      ) -> None:
    
    graphs: List[dict] = [index_data_map[i]['metadata']['graph'] for i in indices]
    results: List[dict] = model.forward_graphs(graphs)
    
    values_true = [graph['graph_labels'] for graph in graphs]
    values_pred = [result['graph_output'] for result in results]
    
    e[f'values/{key}/true'] = values_true
    e[f'values/{key}/pred'] = values_pred
    
    if e.DATASET_TYPE == 'regression':
        
        # prediction error
        r2_value = r2_score(values_true, values_pred)
        mae_value = mean_absolute_error(values_true, values_pred)
        
        e[f'metrics/{key}/r2'] = r2_value
        e[f'metrics/{key}/mae'] = mae_value
        
        e.log(f'regression - r2: {r2_value:.3f} - mae: {mae_value:.3f}')
        
        fig, ax_reg = plt.subplots(figsize=(e.FIG_SIZE, e.FIG_SIZE))
        ax_reg.set_title(f'Regression Fit - {key}\n'
                         f'R2: {r2_value:.3f} - MAE: {mae_value:.3f}')
        plot_regression_fit(values_true, values_pred, ax=ax_reg, num=50)
        e.commit_fig(f'regression_{key}.pdf', fig)
        
        # ~ uncertainty estimation
        errors = [np.abs(true - pred) for true, pred in zip(values_true, values_pred)]
        uncertainties = [result['graph_uncertainty'] for result in results]
        e[f'values/{key}/sigma'] = uncertainties
        
        df = pd.DataFrame({
            'error':            errors,
            'uncertainty':      uncertainties,
        })
        
    elif e.DATASET_TYPE == 'classification':
        
        # classification
        values_true = np.array(values_true)
        values_pred = np.array(values_pred)
        values_pred = [result['graph_probas'] for result in results]
        
        labels_true = np.argmax(values_true, axis=1)
        labels_pred = np.argmax(values_pred, axis=1)
        
        accuracy = accuracy_score(labels_true, labels_pred)
        f1 = f1_score(labels_true, labels_pred, average='macro')
        
        e[f'metrics/{key}/accuracy'] = accuracy
        e[f'metrics/{key}/f1'] = f1
        
        e.log(f'classification - accuracy: {accuracy:.3f} - f1: {f1:.3f}')
        
        # uncertainty estimation
        errors = np.abs(values_true - values_pred).mean(axis=1)
        
        errors = np.array([np.abs(1 - vp[lt]) for vp, lt in zip(values_pred, labels_true)])
        _uncertainties = np.array([result['graph_uncertainty'] for result in results])
        #uncertainties = np.array([uc[lt] for uc, lt in zip(_uncertainties, labels_true)])
        uncertainties = np.array([result['graph_uncertainty'][0] for result in results])
        e[f'values/{key}/sigma'] = uncertainties
        
        df = pd.DataFrame({
            'error':            errors,
            'uncertainty':      uncertainties,
        })
        
        values_true = np.array([vt[lt] for vt, lt in zip(values_true, labels_true)])
        values_pred = np.array([vp[lt] for vp, lt in zip(values_pred, labels_true)])

    # ~ uncertainty error correlation
    corr_value = df['uncertainty'].corr(df['error'])
    e[f'metrics/{key}/corr'] = corr_value
    
    fig, ax_corr = plt.subplots(figsize=(e.FIG_SIZE, e.FIG_SIZE))
    ax_corr.set_title(f'Uncertainty Error Correlation - {key}\n'
                        f'$\\rho$: {corr_value:.3f}')
    # plotting uncertainty versus model error
    ax_corr.scatter(uncertainties, errors, color='red', alpha=0.1, linewidths=0)
    ax_corr.set_xlabel('Uncertainty')
    ax_corr.set_ylabel('Error')
    e.commit_fig(f'uncertainty_error_correlation_{key}.pdf', fig)
    
    # ~ uncertainty error reduction curve
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)
    ths_mean, rds_mean = threshold_error_reduction(
        uncertainties=uncertainties,
        errors=errors,
        num_bins=25,
        error_func=np.mean,
    )
    auc_mean = auc(ths_mean, rds_mean)
    
    ths_max, rds_max = threshold_error_reduction(
        uncertainties=uncertainties,
        errors=errors,
        num_bins=25,
        error_func=np.max,
    )
    auc_max = auc(ths_max, rds_max)
    
    e[f'metrics/{key}/uer_auc_mean'] = auc_mean
    e[f'metrics/{key}/uer_auc_max'] = auc_max
    
    fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(2 * e.FIG_SIZE, e.FIG_SIZE))
    ax_mean.set_title(f'Uncertainty Error Reduction - Mean\n'
                      f'AUC: {auc_mean:.3f}')
    plot_threshold_error_reductions(
        ax=ax_mean, 
        thresholds=ths_mean, 
        reductions=rds_mean,
        color='orange'
    )
    
    ax_max.set_title(f'Uncertainty Error Reduction - Max\n'
                     f'AUC: {auc_max:.3f}')
    plot_threshold_error_reductions(
        ax=ax_max, 
        thresholds=ths_max, 
        reductions=rds_max,
        color='purple'
    )
    e.commit_fig(f'uncertainty_error_reduction_{key}.pdf', fig)
    
    # ~ relative log likelihood
    values_true = np_array(values_true)
    values_pred = np_array(values_pred)
    rll_value = rll_score(values_true, values_pred, uncertainties)
    
    e.log(f'uncertainty'
          f' - corr: {corr_value:.3f}'
          f' - auc_mean: {auc_mean:.3f}'
          f' - auc_max: {auc_max:.3f}'
          f' - rll: {rll_value:.3f}')
        
    # ~ example graphs
    e.log('visualizing the example graphs...')
    example_indices: List[int] = e['indices/example']
    graphs_example = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    example_infos: List[dict] = model.forward_graphs(graphs_example)
    create_importances_pdf(
        graph_list=graphs_example,
        image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
        node_positions_list=[graph['node_positions'] for graph in graphs_example],
        importances_map={
            'megan': (
                [info['node_importance'] for info in example_infos],
                [info['edge_importance'] for info in example_infos],
            )
        },
        output_path=os.path.join(e.path, f'example_explanations_{key}.pdf'),
        plot_node_importances_cb=plot_node_importances_background,
        plot_edge_importances_cb=plot_edge_importances_background,
    )


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')

    models: List[Megan] = []
    for i in range(NUM_MODELS):
        
        exp: Experiment = Experiment.import_from(
            experiment_path=e.BASE_EXPERIMENT,
            glob=globals(),
        )
        exp.__DEBUG__ = False
        exp.__PREFIX__ = f'{e.__PREFIX__}_member_{i}'
        exp.SEED = e.SEED
        exp.USE_BOOTSTRAPPING = True
        
        exp.EPOCHS = e.EPOCHS
        exp.LEARNING_RATE = e.LEARNING_RATE
        exp.BATCH_SIZE = e.BATCH_SIZE
        exp.NUM_TRAIN = e.NUM_TRAIN
        exp.NUM_VAL = e.NUM_VAL
        exp.NUM_TEST = e.NUM_TEST
        exp.NUM_EXAMPLES = 25
        # - model parameters
        if e.IMPORTANCE_OFFSET is not None:
            exp.IMPORTANCE_OFFSET = e.IMPORTANCE_OFFSET
        if e.REGRESSION_MARGIN is not None:
            exp.REGRESSION_MARGIN = e.REGRESSION_MARGIN
        
        # Finally we force that experiment to actually execute.
        exp.update_parameters_special()
        exp.run()
        
        link_path = os.path.join(e.path, f'model_{i}')
        os.symlink(exp.path, link_path)
        
        model: Megan = exp['_model']
        model.eval()
        
        models.append(model)
        
        pprint(exp['indices/test'], max_length=20)
        
        e.log(f'experiment {i} finished')
        
    e.DATASET_TYPE = exp.DATASET_TYPE
    e.CHANNEL_INFOS = exp.CHANNEL_INFOS
        
    e.log('extracting dataset from experiment...')
    index_data_map = exp['_index_data_map']
    val_indices = exp['indices/val']
    test_indices = exp['indices/test']
    e['indices'] = exp['indices']
    
    e.log(f' * total elements: {len(index_data_map)}')
    e.log(f' * val: {len(val_indices)}')
    e.log(f' * test: {len(test_indices)}')
        
    # ~ Building Ensemble
    # Bundeling the individual models into the ensemble model
    e.log('building the ensmeble model...')
    ensemble: MeganEnsemble = MeganEnsemble(
        models=models,
    )
    
    # ~ evaluating model
    # Evaluating the uncalibrated model on the validation and test set.
    e.log('evaluating the ensemble model on validation set...')
    e.apply_hook(
        'evaluate_ensemble',
        model=ensemble,
        index_data_map=index_data_map,
        indices=val_indices,
        key='val',
    )
    
    e.log('evaluating the ensemble model on test set...')
    e.apply_hook(
        'evaluate_ensemble',
        model=ensemble,
        index_data_map=index_data_map,
        indices=test_indices,
        key='test',
    )
    
    # ~ calibrating model
    e.log('calibrating the ensemble model...')
    e.apply_hook(
        'calibrate_model',
        model=ensemble,
        index_data_map=index_data_map,
        indices=val_indices,
    )
    
    # ~ evaluating calibrated model
    e.log('evaluating the calibrated ensemble model on test set...')
    e.apply_hook(
        'evaluate_ensemble',
        model=ensemble,
        index_data_map=index_data_map,
        indices=test_indices,
        key='test_cal',
    )
    
    # ~ saving the model
    e.log('saving the ensemble model...')
    ensemble_path = os.path.join(e.path, 'ensemble.ckpt')
    ensemble.save(ensemble_path)

experiment.run_if_main()