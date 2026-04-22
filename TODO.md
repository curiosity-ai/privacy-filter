# TODO

- [ ] Initialize .NET solution and projects (Core, Console, Tests).
- [ ] Create `.gitignore` for .NET development.
- [ ] Create a new `README.md`.
- [ ] Implement HuggingFace model downloader.
- [ ] Implement `safetensors` loader for model weights.
- [ ] Implement Tokenizer (using `Microsoft.ML.Tokenizers` or similar).
- [ ] Implement Tensor operations and basic math functions with SIMD.
- [ ] Implement `RMSNorm` layer.
- [ ] Implement `RotaryEmbedding` (RoPE).
- [ ] Implement `AttentionBlock` (Grouped-Query Attention).
- [ ] Implement `MLPBlock` (Sparse Mixture-of-Experts with Top-K routing).
- [ ] Implement `TransformerBlock`.
- [ ] Implement `Transformer` model (embedding, blocks, token-classification head).
- [ ] Implement decoding logic (Constrained Viterbi decoding for BIOES spans).
- [ ] Instrument Python code to dump intermediate outputs for validation.
- [ ] Write unit tests to compare C# layer outputs against Python layer outputs.
- [ ] Implement Console application for end-to-end testing.
