import json
import torch
import torch.nn.functional as F
import math

# Set random seed for reproducibility
torch.manual_seed(42)

def generate_rmsnorm_data():
    seq_len = 4
    hidden_size = 128
    eps = 1e-5

    x = torch.randn(seq_len, hidden_size)
    weight = torch.randn(hidden_size)

    # RMSNorm
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps) * weight

    with open("Opf.Tests/rmsnorm_data.json", "w") as f:
        json.dump({
            "input": x.flatten().tolist(),
            "weight": weight.tolist(),
            "eps": eps,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "output": out.flatten().tolist()
        }, f)

def apply_rotary_emb(q, k, seq_len, head_dim, base=10000.0):
    # This matches the C# logic for rotary embeddings
    # q: [seq_len, num_heads, head_dim]
    # k: [seq_len, num_kv_heads, head_dim]

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq) # [seq_len, head_dim/2]

    emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, head_dim]

    cos = emb.cos() # [seq_len, head_dim]
    sin = emb.sin() # [seq_len, head_dim]

    def rotate_half(x):
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        return torch.cat((-x2, x1), dim=-1)

    # We apply RoPE per token per head
    # q is [seq_len, num_heads, head_dim] -> [seq_len, 1, head_dim] to broadcast
    q_out = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_out = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

    return q_out, k_out

def generate_gqa_data():
    seq_len = 5
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads # 32

    x = torch.randn(seq_len, hidden_size)
    wq = torch.randn(hidden_size, num_heads * head_dim)
    wk = torch.randn(hidden_size, num_kv_heads * head_dim)
    wv = torch.randn(hidden_size, num_kv_heads * head_dim)
    wo = torch.randn(num_heads * head_dim, hidden_size)

    # Q, K, V
    q = torch.matmul(x, wq) # [seq_len, num_heads * head_dim]
    k = torch.matmul(x, wk) # [seq_len, num_kv_heads * head_dim]
    v = torch.matmul(x, wv) # [seq_len, num_kv_heads * head_dim]

    q = q.view(seq_len, num_heads, head_dim)
    k = k.view(seq_len, num_kv_heads, head_dim)
    v = v.view(seq_len, num_kv_heads, head_dim)

    # RoPE
    q, k = apply_rotary_emb(q, k, seq_len, head_dim)

    # Attention
    q = q.transpose(0, 1) # [num_heads, seq_len, head_dim]
    k = k.transpose(0, 1) # [num_kv_heads, seq_len, head_dim]
    v = v.transpose(0, 1) # [num_kv_heads, seq_len, head_dim]

    # Repeat KV for GQA
    num_groups = num_heads // num_kv_heads
    k = torch.repeat_interleave(k, num_groups, dim=0) # [num_heads, seq_len, head_dim]
    v = torch.repeat_interleave(v, num_groups, dim=0) # [num_heads, seq_len, head_dim]

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # [num_heads, seq_len, seq_len]
    attn = F.softmax(scores, dim=-1) # [num_heads, seq_len, seq_len]

    out = torch.matmul(attn, v) # [num_heads, seq_len, head_dim]
    out = out.transpose(0, 1).contiguous().view(seq_len, num_heads * head_dim) # [seq_len, num_heads * head_dim]

    final_out = torch.matmul(out, wo) # [seq_len, hidden_size]

    with open("Opf.Tests/gqa_data.json", "w") as f:
        json.dump({
            "input": x.flatten().tolist(),
            "wq": wq.flatten().tolist(),
            "wk": wk.flatten().tolist(),
            "wv": wv.flatten().tolist(),
            "wo": wo.flatten().tolist(),
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "output": final_out.flatten().tolist()
        }, f)

def generate_moe_data():
    seq_len = 3
    hidden_size = 64
    intermediate_size = 128
    num_experts = 4
    top_k = 2

    x = torch.randn(seq_len, hidden_size)
    gate_weight = torch.randn(hidden_size, num_experts)

    expert_w1 = torch.randn(num_experts, hidden_size, intermediate_size)
    expert_w2 = torch.randn(num_experts, intermediate_size, hidden_size)
    expert_w3 = torch.randn(num_experts, hidden_size, intermediate_size)

    # Forward pass
    routing_logits = torch.matmul(x, gate_weight) # [seq_len, num_experts]
    routing_weights, selected_experts = torch.topk(routing_logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1) # [seq_len, top_k]

    final_output = torch.zeros(seq_len, hidden_size)

    for i in range(seq_len):
        for j in range(top_k):
            expert_idx = selected_experts[i, j].item()
            weight = routing_weights[i, j].item()

            token_in = x[i] # [hidden_size]

            h1 = torch.matmul(token_in, expert_w1[expert_idx])
            h3 = torch.matmul(token_in, expert_w3[expert_idx])

            # OPF SwiGLU: x * sigmoid(1.702 * x) * (y + 1)
            # We apply limit clamping logic explicitly:
            limit = 7.0
            h1_clamp = h1.clamp(min=None, max=limit)
            h3_clamp = h3.clamp(min=-limit, max=limit)
            silu_h1 = h1_clamp * torch.sigmoid(1.702 * h1_clamp)
            h_swiglu = silu_h1 * (h3_clamp + 1.0)

            h_out = torch.matmul(h_swiglu, expert_w2[expert_idx])

            final_output[i] += h_out * weight

    with open("Opf.Tests/moe_data.json", "w") as f:
        json.dump({
            "input": x.flatten().tolist(),
            "gate_weight": gate_weight.flatten().tolist(),
            "expert_w1": [e.flatten().tolist() for e in expert_w1],
            "expert_w2": [e.flatten().tolist() for e in expert_w2],
            "expert_w3": [e.flatten().tolist() for e in expert_w3],
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "output": final_output.flatten().tolist()
        }, f)

def generate_viterbi_data():
    seq_len = 4
    num_labels = 3

    logits = torch.randn(seq_len, num_labels)
    start_transitions = torch.randn(num_labels)
    transitions = torch.randn(num_labels, num_labels) # [from, to] (Python code usually puts it like this)
    # The OPF codebase (ViterbiCRFDecoder.Decode) uses score = scores[j] + transitions[j * _numLabels + i]
    # which means transitions is [from * num_labels + to]
    end_transitions = torch.randn(num_labels)

    # Pure Python Viterbi matching C# implementation
    path = torch.zeros(seq_len, num_labels, dtype=torch.long)
    scores = start_transitions + logits[0]

    for t in range(1, seq_len):
        next_scores = torch.zeros(num_labels)
        for i in range(num_labels):
            max_score = float('-inf')
            max_idx = 0
            for j in range(num_labels):
                score = scores[j] + transitions[j, i]
                if score > max_score:
                    max_score = score
                    max_idx = j
            next_scores[i] = max_score + logits[t, i]
            path[t, i] = max_idx
        scores = next_scores

    best_final_score = float('-inf')
    best_last_state = 0
    for i in range(num_labels):
        score = scores[i] + end_transitions[i]
        if score > best_final_score:
            best_final_score = score
            best_last_state = i

    result = [0] * seq_len
    result[-1] = best_last_state
    for t in range(seq_len - 1, 0, -1):
        result[t - 1] = path[t, result[t]].item()

    with open("Opf.Tests/viterbi_data.json", "w") as f:
        json.dump({
            "logits": logits.flatten().tolist(),
            "start_transitions": start_transitions.tolist(),
            "transitions": transitions.flatten().tolist(),
            "end_transitions": end_transitions.tolist(),
            "seq_len": seq_len,
            "num_labels": num_labels,
            "output": result
        }, f)

generate_rmsnorm_data()
generate_gqa_data()
generate_moe_data()
generate_viterbi_data()
print("Artifacts generated successfully.")
