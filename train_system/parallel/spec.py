from dataclasses import dataclass


@dataclass
class OpShardRule:
    module_path: str            # "layers.0.attn.qkv"
    shard_style: str            # "col | row | seq | expert | replicate"
    shard_axis: str = "none"    # "in | out | seq | expert"
    pre_comm: str = "none"      # "all_reduce | all_gather | reduce_scatter | all2all | send/recv"
    post_comm: str = "none"     # "all_reduce | all_gather | reduce_scatter | all2all | send/recv"
    comm_dim: int = -1          # which dim to comm


@dataclass
class ModelParallelSpec:
    rules: list[OpShardRule]
    tie_rules: list[tuple[str, str]]  # e.g. embedding <-> lm_head