from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


experiment = Experiment.extend(
    'vgd_torch.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()