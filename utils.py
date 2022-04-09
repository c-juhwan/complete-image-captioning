import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F

def check_path(path:str):
    """
    Check if the path exists and create it if not.
    """
    if not os.path.exists(path):
        os.mkdir(path)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)

def get_tb_exp_name(args:argparse.Namespace):
    """
    Get the experiment name for tensorboard experiment.
    """

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    exp_name = str()
    exp_name += "%s - " % args.model_name

    if args.training:
        exp_name += 'TRAIN - '
        exp_name += "BS=%i_" % args.batch_size 
        exp_name += "EP=%i_" % args.num_epochs
        exp_name += "LR=%.4f_" % args.learning_rate
    elif args.testing:
        exp_name += 'TEST - '
        exp_name += "BS=%i_" % args.batch_size
    exp_name += "TS=%s" % ts

    return exp_name

def set_random_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)