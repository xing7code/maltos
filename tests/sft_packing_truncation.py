"""split_example_to_seq_len: truncate, and refuse a segment with nothing to learn from."""
import sys; sys.path.insert(0, ".")
from utils.sft_packing import split_example_to_seq_len, EncodedSFTExample as E

SEQ = 8
cases = [
    ("fits untouched",      E([1]*6, [False]*3 + [True]*3),            1, 6,  3),
    ("truncated, keeps targets", E([1]*20, [False]*5 + [True]*15),     1, 8,  3),
    ("truncated, targets all past the window", E([1]*20, [False]*10 + [True]*10), 0, 0, 0),
    ("exactly seq_len",     E([1]*8, [False]*4 + [True]*4),            1, 8,  4),
]
bad = 0
for name, ex, n_exp, len_exp, sup_exp in cases:
    out = split_example_to_seq_len(ex, seq_len=SEQ)
    n = len(out)
    ln = len(out[0].token_ids) if out else 0
    sup = sum(out[0].supervised_mask) if out else 0
    ok = (n, ln, sup) == (n_exp, len_exp, sup_exp)
    bad += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name:<42} segments={n} len={ln} supervised={sup}")
print("\nPASS" if not bad else f"\n{bad} FAILED")
sys.exit(1 if bad else 0)
