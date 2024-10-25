import torch
import numpy as np
import random 
import os


def set_random_seed(seed: int, deterministic: bool):
    """Set seeds"""
    random.seed(seed)
    print(f"Set: random.seed({seed})")
    np.random.seed(seed)
    print(f"Set: np.random.seed({seed})")
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Set: os.environ['PYTHONHASHSEED'] = str({seed})")
    torch.manual_seed(seed)
    print(f"Set: torch.manual_seed({seed})")
    torch.cuda.manual_seed_all(seed)  # type: ignore
    print(f"Set: torch.cuda.manual_seed({seed})")
    torch.backends.cudnn.benchmark = False
    print("Set: torch.backends.cudnn.benchmark = False")
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    print(f"Set: torch.backends.cudnn.deterministic = {deterministic}")
