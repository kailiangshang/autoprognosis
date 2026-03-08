# stdlib
import os
import sys
import warnings

# third party
import optuna

# autoprognosis relative
from . import logger  # noqa: F401

optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


logger.add(sink=sys.stderr, level="INFO")

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pandas\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"optuna\..*")

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
