import torch
SEQ_LEN = 250
K = 10
MAX_T = 20
NUM_BLOCKS = (SEQ_LEN - MAX_T) // K + 1
print(f"NUM_BLOCKS = {NUM_BLOCKS}")
for b in range(NUM_BLOCKS):
    start = b * K
    end = start + MAX_T
print(f"Last block start: {start}, end: {end}")
