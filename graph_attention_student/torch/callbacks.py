"""
PyTorch Lightning callbacks for MEGAN model training.
"""
import signal
import time
import threading
import typing as t
from collections import defaultdict
from typing import List, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    accuracy_score = None
    f1_score = None

# Optional hardware monitoring dependencies
# nvidia-ml-py and pynvml both expose the 'pynvml' module
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings('ignore', category=FutureWarning)
        import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False


class GracefulStopCallback(Callback):
    """
    Catches Ctrl+C (SIGINT) and gracefully stops training instead of killing the process.

    On the first Ctrl+C, sets ``trainer.should_stop = True`` so the current epoch finishes
    and training exits cleanly — allowing evaluation and model saving to proceed.
    On the second Ctrl+C, raises KeyboardInterrupt to force-quit immediately.

    The original signal handler is restored when training ends (via ``teardown``).
    """

    def __init__(self):
        super().__init__()
        self._original_handler = None
        self._trainer = None
        self._interrupted = False

    def on_train_start(self, trainer: pl.Trainer, pl_module):
        self._trainer = trainer
        self._interrupted = False
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        if self._interrupted:
            # Second Ctrl+C — force quit
            print('\nForce quit requested. Exiting immediately.')
            if self._original_handler:
                signal.signal(signal.SIGINT, self._original_handler)
            raise KeyboardInterrupt
        self._interrupted = True
        print('\nGraceful stop requested — finishing current epoch. Press Ctrl+C again to force quit.')
        if self._trainer is not None:
            self._trainer.should_stop = True

    def teardown(self, trainer, pl_module, stage=None):
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None


class ImportanceFactorWarmup(Callback):
    """
    A PyTorch Lightning callback that gradually ramps up the importance factor of a MEGAN model
    over the course of training using a linear schedule.

    The callback reads the target importance factor from the model at the start of training,
    then linearly interpolates from `start_value` to that target over `warmup_epochs` epochs.
    After the warmup period, the importance factor remains at the target value.

    :param start_value: The initial importance factor value at epoch 0. Default is 1e-6.
    :param warmup_epochs: The number of epochs over which to ramp up to the final value.

    Example usage::

        from graph_attention_student.torch.callbacks import ImportanceFactorWarmup

        model = Megan(importance_factor=1.0, ...)
        callback = ImportanceFactorWarmup(start_value=1e-6, warmup_epochs=50)
        trainer = Trainer(callbacks=[callback], max_epochs=150)
        trainer.fit(model, datamodule)
    """

    def __init__(self,
                 start_value: float = 1e-6,
                 warmup_epochs: int = 50,
                 ):
        super().__init__()
        self.start_value = start_value
        self.warmup_epochs = warmup_epochs
        self.final_value: t.Optional[float] = None

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        Called at the start of training. Reads the target importance factor from the model
        and sets the initial value.
        """
        self.final_value = pl_module.importance_factor
        pl_module.importance_factor = self.start_value

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """
        Called at the start of each training epoch. Updates the importance factor
        based on the current epoch using linear interpolation.
        """
        current_epoch = trainer.current_epoch

        if current_epoch >= self.warmup_epochs:
            # Warmup complete, use final value
            pl_module.importance_factor = self.final_value
        else:
            # Linear interpolation
            progress = current_epoch / self.warmup_epochs
            pl_module.importance_factor = (
                self.start_value + progress * (self.final_value - self.start_value)
            )


# Module groups for per-module gradient/weight tracking.
# Keys are display names, values are attribute name prefixes on the Megan model.
MODULE_GROUPS = {
    'Encoder': 'encoder_layers',
    'Channel Proj': 'channel_projection_layers',
    'Contrastive Proj': 'projection_layers',
    'Prediction': 'dense_layers',
    'Importance': 'importance_layers',
}

# Colors for module groups in plots
MODULE_COLORS = {
    'Encoder': '#2196F3',
    'Channel Proj': '#4CAF50',
    'Contrastive Proj': '#FF9800',
    'Prediction': '#9C27B0',
    'Importance': '#F44336',
}


class MeganTrainingMetricsCallback(Callback):
    """
    Comprehensive training metrics callback for MEGAN models.

    Produces a 6x5 matplotlib grid each epoch covering:
    - Row 1: Loss components
    - Row 2: Loss dynamics & training curriculum
    - Row 3: Validation prediction quality
    - Row 4: Explanation quality
    - Row 5: Gradient health & convergence
    - Row 6: Hardware utilization

    :param experiment: PyComex Experiment instance for tracking
    :param val_graphs: Validation graphs for per-epoch evaluation
    :param num_fidelity_samples: Number of graphs for fidelity computation (default 100)
    :param smoothing_window: Window size for moving average smoothing (default 10)
    :param dataset_type: 'regression' or 'classification'
    :param num_channels: Number of explanation channels
    :param channel_infos: Dict mapping channel index to name/color info
    """

    def __init__(self,
                 experiment,
                 val_graphs: list,
                 num_fidelity_samples: int = 100,
                 smoothing_window: int = 10,
                 dataset_type: str = 'regression',
                 num_channels: int = 2,
                 channel_infos: Optional[dict] = None,
                 ):
        super().__init__()
        self.experiment = experiment
        self.val_graphs = val_graphs
        self.num_fidelity_samples = min(num_fidelity_samples, len(val_graphs))
        self.smoothing_window = smoothing_window
        self.dataset_type = dataset_type
        self.num_channels = num_channels
        self.channel_infos = channel_infos or {
            0: {'name': 'negative', 'color': 'skyblue'},
            1: {'name': 'positive', 'color': 'coral'},
        }

        # -- Epoch-level persistent lists --
        # Losses
        self.train_losses: List[float] = []
        self.pred_losses: List[float] = []
        self.expl_losses: List[float] = []
        self.cont_losses: List[float] = []
        self.fid_losses: List[float] = []
        self.spar_losses: List[float] = []
        self.unif_losses: List[float] = []
        self.uniformity: List[float] = []

        # Loss dynamics
        self.importance_factors: List[float] = []
        self.contrastive_factors: List[float] = []
        self.learning_rates: List[float] = []

        # Contrastive
        self.sim_pos_values: List[float] = []
        self.sim_neg_values: List[float] = []

        # Validation - prediction
        self.primary_metric: List[float] = []   # R2 or Accuracy
        self.secondary_metric: List[float] = [] # MAE or F1
        self.approx_values: List[float] = []
        self.per_channel_approx: List[list] = []
        self.approx_auc: List[float] = []
        self.per_channel_auc: List[list] = []

        # Explanation quality
        self.importance_sparsity: List[np.ndarray] = []
        self.fidelity_values: List[np.ndarray] = []
        self.fidelity_sign_consistency: List[np.ndarray] = []
        # ~ explanation-loss redesign metrics
        # These terms are invisible in the original grid, which predates them. Two of the failures
        # this session cost several runs each precisely because the relevant quantity was not
        # plotted anywhere: a mask that is structurally correct but numerically collapsed, and a
        # margin constant that is either unreachable or inert depending on the dataset.
        self.rdt_losses: List[float] = []
        self.polar_losses: List[float] = []
        self.spread_losses: List[float] = []
        self.margin_met: List[float] = []
        # mask health, computed from the validation graphs each epoch
        self.mask_raw_max: List[float] = []
        self.mask_empty_frac: List[np.ndarray] = []
        self.mask_midband: List[float] = []
        self.latest_node_importances: Optional[List[np.ndarray]] = None
        self.latest_edge_importances: Optional[List[np.ndarray]] = None
        self.latest_values_true: Optional[np.ndarray] = None
        self.latest_values_pred: Optional[np.ndarray] = None

        # Gradients & convergence
        self.global_grad_norms: List[float] = []
        self.module_grad_norms: Dict[str, List[float]] = defaultdict(list)
        self.param_deltas: List[float] = []
        self.module_param_deltas: Dict[str, List[float]] = defaultdict(list)
        self.weight_norms: Dict[str, List[float]] = defaultdict(list)

        # Hardware
        self.gpu_utilization: List[float] = []
        self.gpu_memory: List[float] = []
        self.cpu_utilization: List[float] = []
        self.cpu_ram: List[float] = []
        self.epoch_durations: List[float] = []

        # -- Per-epoch accumulators (reset each epoch) --
        self._epoch_grad_norms: List[float] = []
        self._epoch_module_grad_norms: Dict[str, List[float]] = defaultdict(list)
        self._epoch_losses: Dict[str, List[float]] = defaultdict(list)
        self._param_snapshot: Optional[dict] = None
        self._epoch_start_time: float = 0.0

        # Hardware monitoring thread state
        self._hw_thread: Optional[threading.Thread] = None
        self._hw_stop_event = threading.Event()
        self._hw_lock = threading.Lock()
        self._hw_samples: List[dict] = []
        self._nvml_handle = None

        if _NVML_AVAILABLE:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    # ---------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------

    def _smooth(self, values: list, window: Optional[int] = None) -> list:
        """Causal (left-looking) simple moving average."""
        window = window or self.smoothing_window
        if len(values) < 2:
            return list(values)
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
        return smoothed

    def _compute_gradient_norm(self, model, prefix=None) -> float:
        norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if prefix is None or name.startswith(prefix):
                    norms.append(param.grad.data.norm(2))
        if not norms:
            return 0.0
        return torch.stack(norms).pow(2).sum().sqrt().item()

    def _compute_weight_norm(self, model, prefix) -> float:
        norms = []
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                norms.append(param.data.norm(2))
        if not norms:
            return 0.0
        return torch.stack(norms).pow(2).sum().sqrt().item()

    def _take_param_snapshot(self, model) -> dict:
        return {name: param.data.clone().cpu()
                for name, param in model.named_parameters()}

    def _compute_param_delta(self, model, snapshot, prefix=None) -> float:
        total_delta = 0.0
        for name, param in model.named_parameters():
            if prefix is None or name.startswith(prefix):
                if name in snapshot:
                    delta = (param.data.cpu() - snapshot[name]).norm(2).item() ** 2
                    total_delta += delta
        return total_delta ** 0.5

    def _reset_epoch_accumulators(self):
        self._epoch_grad_norms = []
        self._epoch_module_grad_norms = defaultdict(list)
        self._epoch_losses = defaultdict(list)

    # ---------------------------------------------------------------
    # Hardware monitoring
    # ---------------------------------------------------------------

    def _start_hw_monitor(self):
        self._hw_stop_event.clear()
        self._hw_samples = []
        # Prime psutil CPU counter
        if _PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
        self._hw_thread = threading.Thread(target=self._hw_sample_loop, daemon=True)
        self._hw_thread.start()

    def _hw_sample_loop(self):
        while not self._hw_stop_event.is_set():
            sample = {}
            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    sample['gpu_util'] = float(util.gpu)
                    sample['gpu_mem_mb'] = mem_info.used / (1024 ** 2)
                except Exception:
                    pass
            if _PSUTIL_AVAILABLE:
                try:
                    sample['cpu_util'] = psutil.cpu_percent(interval=None)
                    sample['ram_mb'] = psutil.virtual_memory().used / (1024 ** 2)
                except Exception:
                    pass
            if sample:
                with self._hw_lock:
                    self._hw_samples.append(sample)
            self._hw_stop_event.wait(timeout=1.0)

    def _stop_hw_monitor(self) -> dict:
        if self._hw_thread is None:
            return {}
        self._hw_stop_event.set()
        self._hw_thread.join(timeout=3.0)
        with self._hw_lock:
            samples = list(self._hw_samples)
            self._hw_samples.clear()
        result = {}
        for key in ('gpu_util', 'gpu_mem_mb', 'cpu_util', 'ram_mb'):
            values = [s[key] for s in samples if key in s]
            if values:
                result[key] = sum(values) / len(values)
        return result

    # ---------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module):
        self._reset_epoch_accumulators()
        self._param_snapshot = self._take_param_snapshot(pl_module)
        self._epoch_start_time = time.time()
        self._start_hw_monitor()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grad_norm = self._compute_gradient_norm(pl_module)
        self._epoch_grad_norms.append(grad_norm)
        for group_name, prefix in MODULE_GROUPS.items():
            module_grad = self._compute_gradient_norm(pl_module, prefix=prefix)
            self._epoch_module_grad_norms[group_name].append(module_grad)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = getattr(pl_module, 'batch_metrics', {})
        if not metrics:
            return
        for key in ('loss', 'loss_pred', 'loss_expl', 'loss_spar', 'loss_cont', 'loss_unif',
                    'loss_fid', 'loss_rdt', 'loss_polar', 'loss_spread', 'margin_met'):
            if key in metrics:
                self._epoch_losses[key].append(float(metrics[key]))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        epoch_duration = time.time() - self._epoch_start_time
        self.epoch_durations.append(epoch_duration)

        # -- Aggregate losses --
        for key, target_list in [
            ('loss', self.train_losses),
            ('loss_pred', self.pred_losses),
            ('loss_expl', self.expl_losses),
            ('loss_spar', self.spar_losses),
            ('loss_cont', self.cont_losses),
            ('loss_unif', self.unif_losses),
            ('loss_fid', self.fid_losses),
            ('loss_rdt', self.rdt_losses),
            ('loss_polar', self.polar_losses),
            ('loss_spread', self.spread_losses),
            ('margin_met', self.margin_met),
        ]:
            vals = self._epoch_losses.get(key, [])
            target_list.append(np.mean(vals) if vals else 0.0)

        # -- Loss dynamics --
        self.importance_factors.append(float(getattr(pl_module, 'importance_factor', 0.0)))
        self.contrastive_factors.append(float(getattr(pl_module, 'contrastive_factor', 0.0)))
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0
        self.learning_rates.append(float(lr))

        # -- Contrastive similarities from trainer metrics --
        sim_pos = trainer.callback_metrics.get('sim_pos')
        self.sim_pos_values.append(float(sim_pos) if sim_pos is not None else 0.0)
        sim_neg = trainer.callback_metrics.get('sim_neg')
        self.sim_neg_values.append(float(sim_neg) if sim_neg is not None else 0.0)

        # -- Embedding uniformity from queue --
        # ||mean(queue)|| for unit-normalized embeddings: 1.0 = collapsed, 0.0 = uniform
        concentrations = []
        for k in range(self.num_channels):
            queue = getattr(pl_module, f'unif_queue_{k}', None)
            if queue is not None:
                mean_vec = queue.mean(dim=1)  # (D,) — mean across queue entries
                concentrations.append(float(mean_vec.norm().cpu()))
        self.uniformity.append(np.mean(concentrations) if concentrations else 1.0)

        # -- Gradients --
        if self._epoch_grad_norms:
            self.global_grad_norms.append(np.mean(self._epoch_grad_norms))
        else:
            self.global_grad_norms.append(0.0)
        for group_name in MODULE_GROUPS:
            vals = self._epoch_module_grad_norms.get(group_name, [])
            self.module_grad_norms[group_name].append(np.mean(vals) if vals else 0.0)

        # -- Parameter deltas & weight norms --
        if self._param_snapshot is not None:
            self.param_deltas.append(self._compute_param_delta(pl_module, self._param_snapshot))
            for group_name, prefix in MODULE_GROUPS.items():
                self.module_param_deltas[group_name].append(
                    self._compute_param_delta(pl_module, self._param_snapshot, prefix=prefix)
                )
        for group_name, prefix in MODULE_GROUPS.items():
            self.weight_norms[group_name].append(self._compute_weight_norm(pl_module, prefix))

        # -- Hardware --
        hw = self._stop_hw_monitor()
        self.gpu_utilization.append(hw.get('gpu_util', 0.0))
        self.gpu_memory.append(hw.get('gpu_mem_mb', 0.0))
        self.cpu_utilization.append(hw.get('cpu_util', 0.0))
        self.cpu_ram.append(hw.get('ram_mb', 0.0))

        # -- Validation (run on CPU, eval mode) --
        device = pl_module.device
        pl_module.eval()
        pl_module.to('cpu')

        try:
            self._run_validation(pl_module)
        except Exception as exc:
            self.experiment.log(f'Warning: Metrics validation failed: {exc}')

        pl_module.train()
        pl_module.to(device)

        # -- Generate plot --
        try:
            fig = self._create_metrics_plot(epoch)
            self.experiment.track('training_metrics', fig)
            plt.close(fig)
        except Exception as exc:
            self.experiment.log(f'Warning: Failed to create metrics plot: {exc}')

    def teardown(self, trainer, pl_module, stage=None):
        self._stop_hw_monitor()

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    def _run_validation(self, model):
        """Run validation metrics on the validation set."""
        results = model.forward_graphs(self.val_graphs)

        values_true = np.array([g['graph_labels'] for g in self.val_graphs])
        values_pred = np.array([r['graph_output'] for r in results])

        # Store for the latest-epoch regression/classification plot
        self.latest_values_true = values_true
        self.latest_values_pred = values_pred

        # -- Prediction quality --
        if self.dataset_type == 'regression':
            self.primary_metric.append(float(r2_score(values_true, values_pred)))
            self.secondary_metric.append(float(mean_absolute_error(values_true, values_pred)))
        elif self.dataset_type == 'classification' and accuracy_score is not None:
            yt = np.argmax(values_true, axis=1)
            yp = np.argmax(values_pred, axis=1)
            self.primary_metric.append(float(accuracy_score(yt, yp)))
            self.secondary_metric.append(float(f1_score(yt, yp, average='macro')))

        # -- Explanation approximation --
        approx_true, approx_pred = model._predict_approximate(
            results=results, values_true=values_true
        )

        # Per-channel AUC (threshold-free separability) and optimal accuracy
        per_ch_auc = []
        per_ch_acc = []
        for k in range(self.num_channels):
            if k < approx_true.shape[1]:
                yt_k = approx_true[:, k].astype(float)
                yp_k = approx_pred[:, k]

                # AUC — measures separability across all thresholds
                try:
                    if len(np.unique(yt_k)) > 1:
                        auc_k = float(roc_auc_score(yt_k, yp_k))
                    else:
                        auc_k = 0.5
                except ValueError:
                    auc_k = 0.5
                per_ch_auc.append(auc_k)

                # Optimal accuracy — best threshold for this channel
                thresholds = np.unique(yp_k)
                best_acc = 0.5
                for t in thresholds:
                    acc = float(np.mean(yt_k == (yp_k > t).astype(float)))
                    best_acc = max(best_acc, acc)
                per_ch_acc.append(best_acc)

        self.per_channel_auc.append(per_ch_auc)
        self.per_channel_approx.append(per_ch_acc)

        # Aggregate metrics (mean across channels)
        self.approx_auc.append(float(np.mean(per_ch_auc)) if per_ch_auc else 0.5)
        self.approx_values.append(float(np.mean(per_ch_acc)) if per_ch_acc else 0.5)

        # -- Importance statistics --
        node_imps_per_channel = [[] for _ in range(self.num_channels)]
        edge_imps_per_channel = [[] for _ in range(self.num_channels)]
        mean_importance = np.zeros(self.num_channels)
        for r in results:
            ni = r['node_importance']  # (V, K)
            ei = r['edge_importance']  # (E, K)
            for k in range(self.num_channels):
                if k < ni.shape[1]:
                    node_imps_per_channel[k].append(ni[:, k])
                    edge_imps_per_channel[k].append(ei[:, k])
                    mean_importance[k] += np.mean(ni[:, k])
        mean_importance /= max(len(results), 1)
        self.importance_sparsity.append(mean_importance)

        # ~ mask health
        # raw max says whether the reported mask has a usable scale at all - a run can reach recall
        # 0.99 at a threshold of 0.1 while reading 0.083 at 0.5 if this collapses. empty fraction
        # catches masks that vanish entirely, which no aggregate mean reveals. mid-band counts
        # values left undecided between 0.1 and 0.5, which separates a genuinely sparse mask from a
        # merely faint one.
        raw_maxes, empties = [], np.zeros(self.num_channels)
        mid_total, mid_count = 0.0, 0
        for r in results:
            ni = r['node_importance']
            raw_maxes.append(float(ni.max()) if ni.size else 0.0)
            for k in range(min(self.num_channels, ni.shape[1])):
                if not (ni[:, k] > 0.5).any():
                    empties[k] += 1
            if ni.size:
                mid_total += float(np.mean((ni >= 0.1) & (ni <= 0.5)))
                mid_count += 1
        self.mask_raw_max.append(float(np.mean(raw_maxes)) if raw_maxes else 0.0)
        self.mask_empty_frac.append(empties / max(len(results), 1))
        self.mask_midband.append(mid_total / max(mid_count, 1))

        self.latest_node_importances = [np.concatenate(v) if v else np.array([])
                                        for v in node_imps_per_channel]
        self.latest_edge_importances = [np.concatenate(v) if v else np.array([])
                                        for v in edge_imps_per_channel]

        # -- Fidelity (on subset) --
        try:
            subset_indices = np.random.choice(
                len(self.val_graphs), size=self.num_fidelity_samples, replace=False
            )
            subset_graphs = [self.val_graphs[i] for i in subset_indices]
            deviations = model.leave_one_out_deviations(subset_graphs)
            # deviations: (B, O, K)
            mean_dev = np.mean(deviations, axis=(0, 1))  # (K,)
            self.fidelity_values.append(mean_dev)

            # Sign consistency: for regression, ch0 should be negative, ch1 positive
            sign_consistency = np.zeros(self.num_channels)
            for k in range(self.num_channels):
                dev_k = deviations[:, :, k].mean(axis=1)  # (B,)
                if self.dataset_type == 'regression':
                    if k == 0:
                        sign_consistency[k] = float(np.mean(dev_k < 0))
                    else:
                        sign_consistency[k] = float(np.mean(dev_k > 0))
                else:
                    sign_consistency[k] = float(np.mean(dev_k > 0))
            self.fidelity_sign_consistency.append(sign_consistency)
        except Exception:
            self.fidelity_values.append(np.zeros(self.num_channels))
            self.fidelity_sign_consistency.append(np.zeros(self.num_channels))

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------

    def _create_metrics_plot(self, epoch: int):
        fig, axes = plt.subplots(7, 5, figsize=(28, 28))
        fig.suptitle(f'MEGAN Training Metrics — Epoch {epoch}', fontsize=14, fontweight='bold')

        epochs = list(range(len(self.train_losses)))
        if not epochs:
            return fig

        # Row 1: Loss components
        self._plot_loss_line(axes[0, 0], epochs, self.train_losses, 'Total Loss', '#333333')
        self._plot_loss_line(axes[0, 1], epochs, self.pred_losses, 'Prediction Loss', '#2196F3')
        self._plot_loss_line(axes[0, 2], epochs, self.expl_losses, 'Explanation Loss', '#4CAF50')
        self._plot_loss_line(axes[0, 3], epochs, self.cont_losses, 'Contrastive Loss', '#FF9800')
        self._plot_dual_loss(axes[0, 4], epochs, self.fid_losses, self.spar_losses,
                             'Fidelity', 'Sparsity', '#9C27B0', '#F44336')

        # Row 2: Loss dynamics
        self._plot_loss_ratios(axes[1, 0], epochs)
        self._plot_effective_weights(axes[1, 1], epochs)
        self._plot_loss_line(axes[1, 2], epochs, self.sim_pos_values, 'Positive Similarity', '#4CAF50')
        self._plot_loss_line(axes[1, 3], epochs, self.uniformity, 'Uniformity', '#00BCD4')
        self._plot_loss_line(axes[1, 4], epochs, self.learning_rates, 'Learning Rate', '#607D8B')

        # Row 3: Validation prediction quality
        label1 = 'R²' if self.dataset_type == 'regression' else 'Accuracy'
        label2 = 'MAE' if self.dataset_type == 'regression' else 'F1 (macro)'
        self._plot_loss_line(axes[2, 0], epochs, self.primary_metric, label1, '#2196F3')
        self._plot_loss_line(axes[2, 1], epochs, self.secondary_metric, label2, '#FF9800')
        self._plot_loss_line(axes[2, 2], epochs, self.approx_auc,
                             'Explanation AUC', '#4CAF50')
        self._plot_per_channel_auc(axes[2, 3], epochs)
        self._plot_latest_fit(axes[2, 4])

        # Row 4: Explanation quality
        self._plot_importance_sparsity(axes[3, 0], epochs)
        self._plot_fidelity_values(axes[3, 1], epochs)
        self._plot_importance_hist(axes[3, 2], self.latest_node_importances, 'Node Importance Dist')
        self._plot_importance_hist(axes[3, 3], self.latest_edge_importances, 'Edge Importance Dist')
        self._plot_fidelity_sign(axes[3, 4], epochs)

        # Row 5: Gradient health & convergence
        self._plot_loss_line(axes[4, 0], epochs, self.global_grad_norms,
                             'Global Gradient Norm', '#333333')
        self._plot_module_lines(axes[4, 1], epochs, self.module_grad_norms, 'Per-Module Grad Norm')
        self._plot_loss_line(axes[4, 2], epochs, self.param_deltas,
                             'Parameter Delta (total)', '#9C27B0')
        self._plot_module_lines(axes[4, 3], epochs, self.module_param_deltas, 'Per-Module Param Delta')
        self._plot_module_lines(axes[4, 4], epochs, self.weight_norms, 'Weight Norms')

        # Row 6: Hardware
        self._plot_hw(axes[5, 0], epochs, self.gpu_utilization, 'GPU Utilization (%)',
                      '#76b900', ylim=(0, 105))
        self._plot_hw(axes[5, 1], epochs, [v / 1024 for v in self.gpu_memory],
                      'GPU Memory (GB)', '#e67e22')
        self._plot_hw(axes[5, 2], epochs, self.cpu_utilization, 'CPU Utilization (%)',
                      '#3498db', ylim=(0, 105))
        self._plot_hw(axes[5, 3], epochs, [v / 1024 for v in self.cpu_ram],
                      'CPU RAM (GB)', '#e74c3c')
        self._plot_hw(axes[5, 4], epochs, self.epoch_durations, 'Epoch Duration (s)', '#607D8B')

        # -- Row 7: explanation-loss redesign --
        # These panels exist because their absence was expensive. Several runs this session were
        # spent chasing symptoms that any one of them would have shown immediately: a mask that was
        # structurally right but numerically collapsed, and a margin constant that is unreachable on
        # one dataset and inert on another at the same nominal value.
        self._plot_redesign_losses(axes[6, 0], epochs)
        self._plot_loss_line(axes[6, 1], epochs, self.margin_met,
                             'Margin Satisfied (frac)', '#9C27B0')
        self._plot_loss_line(axes[6, 2], epochs, self.mask_raw_max,
                             'Mask Scale (per-graph raw max)', '#E91E63')
        self._plot_empty_fraction(axes[6, 3], epochs)
        self._plot_loss_line(axes[6, 4], epochs, self.mask_midband,
                             'Undecided Values (0.1-0.5)', '#795548')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # -- Individual plot helpers --

    def _plot_redesign_losses(self, ax, epochs):
        """The three redesign loss terms on shared axes, since they are read against each other."""
        series = [
            (self.rdt_losses, 'RDT', '#00897B'),
            (self.polar_losses, 'Polarization', '#F4511E'),
            (self.spread_losses, 'Spread', '#3949AB'),
        ]
        plotted = False
        for values, label, color in series:
            if values and any(abs(v) > 1e-12 for v in values):
                ax.plot(epochs[:len(values)], values, color=color, label=label, linewidth=1.6)
                plotted = True
        ax.set_title('Redesign Losses', fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'inactive', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#999999')

    def _plot_empty_fraction(self, ax, epochs):
        """Fraction of validation graphs whose mask is entirely empty, per channel.

        An aggregate mean cannot show this: a channel can have a healthy average importance while a
        substantial share of individual graphs get no explanation at all.
        """
        if self.mask_empty_frac:
            arr = np.array(self.mask_empty_frac)
            for k in range(arr.shape[1]):
                info = self.channel_infos.get(k, {})
                ax.plot(epochs[:len(arr)], arr[:, k],
                        color=info.get('color', f'C{k}'),
                        label=info.get('name', f'channel {k}'), linewidth=1.6)
            ax.legend(fontsize=7)
            ax.set_ylim(-0.02, 1.02)
        ax.set_title('Empty Masks (frac of graphs)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_loss_line(self, ax, epochs, values, title, color):
        ax.set_title(title, fontsize=9)
        if not values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        n = min(len(epochs), len(values))
        e, v = epochs[:n], values[:n]
        ax.plot(e, v, alpha=0.3, color=color)
        ax.plot(e, self._smooth(v), color=color, linewidth=2)
        ax.grid(True, alpha=0.3)

    def _plot_dual_loss(self, ax, epochs, vals1, vals2, label1, label2, color1, color2):
        ax.set_title(f'{label1} + {label2}', fontsize=9)
        if not vals1 and not vals2:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        for vals, label, color in [(vals1, label1, color1), (vals2, label2, color2)]:
            n = min(len(epochs), len(vals))
            if n > 0:
                e, v = epochs[:n], vals[:n]
                ax.plot(e, v, alpha=0.3, color=color)
                ax.plot(e, self._smooth(v), color=color, linewidth=2, label=label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_loss_ratios(self, ax, epochs):
        ax.set_title('Loss Component Ratios', fontsize=9)
        n = len(epochs)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        components = {
            'Pred': self.pred_losses,
            'Expl': self.expl_losses,
            'Cont': self.cont_losses,
            'Unif': self.unif_losses,
            'Fid': self.fid_losses,
            'Spar': self.spar_losses,
        }
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#00BCD4', '#9C27B0', '#F44336']
        stacked = []
        labels = []
        for (label, vals), color in zip(components.items(), colors):
            v = vals[:n] if len(vals) >= n else vals + [0.0] * (n - len(vals))
            stacked.append(v)
            labels.append(label)
        totals = [max(sum(s[i] for s in stacked), 1e-8) for i in range(n)]
        ratios = [[s[i] / totals[i] for i in range(n)] for s in stacked]
        ax.stackplot(epochs[:n], *ratios, labels=labels, colors=colors, alpha=0.7)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_effective_weights(self, ax, epochs):
        ax.set_title('Effective Loss Weights', fontsize=9)
        n = len(epochs)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        for vals, label, color in [
            (self.importance_factors, 'Importance', '#4CAF50'),
            (self.contrastive_factors, 'Contrastive', '#FF9800'),
        ]:
            m = min(n, len(vals))
            if m > 0:
                ax.plot(epochs[:m], vals[:m], color=color, linewidth=2, label=label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_per_channel_auc(self, ax, epochs):
        ax.set_title('Per-Channel Explanation AUC / Opt. Acc', fontsize=9)
        if not self.per_channel_auc:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        n = min(len(epochs), len(self.per_channel_auc))
        for k in range(self.num_channels):
            color = self.channel_infos.get(k, {}).get('color', f'C{k}')
            name = self.channel_infos.get(k, {}).get('name', f'ch{k}')
            # AUC — solid lines
            auc_vals = [self.per_channel_auc[i][k]
                        for i in range(n) if k < len(self.per_channel_auc[i])]
            if auc_vals:
                ax.plot(epochs[:len(auc_vals)], auc_vals, color=color,
                        linewidth=2, label=f'{name} AUC')
            # Optimal accuracy — dashed lines, low opacity
            if self.per_channel_approx:
                m = min(len(epochs), len(self.per_channel_approx))
                acc_vals = [self.per_channel_approx[i][k]
                            for i in range(m) if k < len(self.per_channel_approx[i])]
                if acc_vals:
                    ax.plot(epochs[:len(acc_vals)], acc_vals, color=color,
                            linewidth=1.5, linestyle='--', alpha=0.4, label=f'{name} opt acc')
        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plot_latest_fit(self, ax):
        """Plot regression scatter or confusion matrix for latest epoch."""
        if self.latest_values_true is None or self.latest_values_pred is None:
            ax.set_title('Latest Prediction', fontsize=9)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        yt = self.latest_values_true
        yp = self.latest_values_pred

        if self.dataset_type == 'regression':
            r2 = self.primary_metric[-1] if self.primary_metric else 0
            mae = self.secondary_metric[-1] if self.secondary_metric else 0
            ax.set_title(f'Regression Fit (R²={r2:.3f}, MAE={mae:.3f})', fontsize=9)
            ax.scatter(yt.flatten(), yp.flatten(), alpha=0.4, s=8, color='#2196F3')
            # Plot diagonal reference line
            vmin = min(yt.min(), yp.min())
            vmax = max(yt.max(), yp.max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel('True', fontsize=8)
            ax.set_ylabel('Predicted', fontsize=8)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.3)
        else:
            acc = self.primary_metric[-1] if self.primary_metric else 0
            ax.set_title(f'Confusion Matrix (Acc={acc:.3f})', fontsize=9)
            yt_cls = np.argmax(yt, axis=1)
            yp_cls = np.argmax(yp, axis=1)
            n_classes = yt.shape[1]
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(yt_cls, yp_cls):
                cm[t, p] += 1
            # Row-normalize for display
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm / (row_sums + 1e-8)
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('True', fontsize=8)

    def _plot_importance_sparsity(self, ax, epochs):
        ax.set_title('Mean Importance / Channel', fontsize=9)
        if not self.importance_sparsity:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        n = min(len(epochs), len(self.importance_sparsity))
        for k in range(self.num_channels):
            vals = [self.importance_sparsity[i][k] for i in range(n)]
            color = self.channel_infos.get(k, {}).get('color', f'C{k}')
            name = self.channel_infos.get(k, {}).get('name', f'ch{k}')
            ax.plot(epochs[:n], vals, color=color, linewidth=2, label=name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_fidelity_values(self, ax, epochs):
        ax.set_title('Mean Fidelity / Channel', fontsize=9)
        if not self.fidelity_values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        n = min(len(epochs), len(self.fidelity_values))
        for k in range(self.num_channels):
            vals = [self.fidelity_values[i][k] for i in range(n)]
            color = self.channel_infos.get(k, {}).get('color', f'C{k}')
            name = self.channel_infos.get(k, {}).get('name', f'ch{k}')
            ax.plot(epochs[:n], vals, color=color, linewidth=2, label=name)
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_importance_hist(self, ax, importances, title):
        ax.set_title(title, fontsize=9)
        if importances is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        for k in range(min(len(importances), self.num_channels)):
            if len(importances[k]) == 0:
                continue
            color = self.channel_infos.get(k, {}).get('color', f'C{k}')
            name = self.channel_infos.get(k, {}).get('name', f'ch{k}')
            ax.hist(importances[k], bins=50, alpha=0.5, color=color, label=name, density=True)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_fidelity_sign(self, ax, epochs):
        ax.set_title('Fidelity Sign Consistency', fontsize=9)
        if not self.fidelity_sign_consistency:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        n = min(len(epochs), len(self.fidelity_sign_consistency))
        for k in range(self.num_channels):
            vals = [self.fidelity_sign_consistency[i][k] for i in range(n)]
            color = self.channel_infos.get(k, {}).get('color', f'C{k}')
            name = self.channel_infos.get(k, {}).get('name', f'ch{k}')
            ax.plot(epochs[:n], vals, color=color, linewidth=2, label=name)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_module_lines(self, ax, epochs, data_dict, title):
        ax.set_title(title, fontsize=9)
        if not data_dict:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        for group_name in MODULE_GROUPS:
            vals = data_dict.get(group_name, [])
            n = min(len(epochs), len(vals))
            if n > 0:
                color = MODULE_COLORS.get(group_name, '#333333')
                ax.plot(epochs[:n], vals[:n], color=color, linewidth=1.5,
                        label=group_name, alpha=0.8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_hw(self, ax, epochs, values, title, color, ylim=None):
        ax.set_title(title, fontsize=9)
        if not values or all(v == 0 for v in values):
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
            return
        n = min(len(epochs), len(values))
        e, v = epochs[:n], values[:n]
        ax.plot(e, v, color=color, linewidth=2)
        ax.fill_between(e, v, alpha=0.15, color=color)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
