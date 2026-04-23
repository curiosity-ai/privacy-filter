import json
import torch
import torch.nn.functional as F

def generate_rmsnorm_artifact():
    torch.manual_seed(42)
    # create a random tensor: batch=2, seq=3, features=4
    x = torch.randn(2, 3, 4)
    # simulate rms norm weights
    scale = torch.tensor([1.0, 1.5, 2.0, 0.5])
    eps = 1e-5

    # manual RMSNorm
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    out = x_normed * scale

    with open("artifacts/rmsnorm_in.json", "w") as f:
        json.dump(x.flatten().tolist(), f)
    with open("artifacts/rmsnorm_out.json", "w") as f:
        json.dump(out.flatten().tolist(), f)

import os
os.makedirs("artifacts", exist_ok=True)
generate_rmsnorm_artifact()

def generate_decoder_artifact():
    torch.manual_seed(42)
    # mock token logprobs: seq_len=5, num_classes=33
    logprobs = torch.randn(5, 33)

    # We will simulate a simple decode logic mirroring the Viterbi output.
    # For a real integration, we'd invoke OPF's ViterbiCRFDecoder directly.
    # To keep dependencies minimal and exact, we just record inputs for the test.
    with open("artifacts/viterbi_in.json", "w") as f:
        json.dump(logprobs.flatten().tolist(), f)

generate_decoder_artifact()
