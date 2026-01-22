# main.py
import os
import sys
import random
import logging
import numpy as np
import torch
import torch.multiprocessing as mp
from ehrsyn.config import ex
from trainer import Trainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


@ex.automain
def main(_config):
    set_seed(_config["seed"])
    trainer = Trainer(_config)
    
    if _config.get("encode_only_mode", False):
        trainer.train()
    elif not _config["test_only"]:
        trainer.train()
    else:
        if _config["sample"]:
            trainer.comprehensive_test()
        else:
            trainer.test()