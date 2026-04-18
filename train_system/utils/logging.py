from absl import logging
import torch.distributed as dist


def debug_log(vlevel, msg, rank=0, allow_non_dist=True):
    if not dist.is_initialized():
        if allow_non_dist:
            logging.vlog(vlevel, msg)
    else:
        if dist.get_rank() == rank:
            logging.vlog(vlevel, msg)
