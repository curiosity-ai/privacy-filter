# TODO

- [x] Create `.gitignore` for .NET.
- [x] Create `README.md` outlining the C# project.
- [x] Initialize .NET solution and projects (Core, Console, Tests).
- [x] Create HuggingFace downloader to fetch `openai/privacy-filter`.
- [x] Implement `Safetensors` loader in C# to parse `model.safetensors` / MXFP4 files.
- [x] Implement Tokenizer (Tiktoken / BPE logic).
- [ ] Implement Tensor operations wrapper using `System.Numerics.Tensors` / SIMD.
- [ ] Implement Transformer model architecture in C#:
  - [ ] Embeddings
  - [ ] Grouped Query Attention (GQA)
  - [x] RoPE (Rotary Positional Embeddings)
  - [ ] Sparse Mixture of Experts (MoE) implementation
  - [ ] RMSNorm
- [ ] Implement Model inference pass (logits generation).
- [ ] Implement Sequence Decoding (Viterbi span decoder).
- [ ] Implement Python verification scripts (dump Python intermediates for testing).
- [ ] Create tests to ensure output parity with Python implementation.
- [ ] Implement Console App logic (redact text, read from stdin, etc.).
