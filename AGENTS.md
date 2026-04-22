# OPF C# Port Goals

The goal of this project is to create a pure, modern C# port of the OpenAI Privacy Filter (OPF) model. This involves translating the PyTorch-based Transformer model into C# using modern features like `System.Numerics.Tensors`, `System.Runtime.Intrinsics`, SIMD, and Memory/Span for high performance.

Key objectives:
- Implement the Transformer model (embeddings, attention, mixture-of-experts MLP, RMSNorm) in C#.
- Provide a HuggingFace model downloader to retrieve `openai/privacy-filter`.
- Parse and load the `safetensors` model weights into C# tensors.
- Implement tokenization (BPE) equivalent to `tiktoken` in C# (or using a C# library like `Microsoft.ML.Tokenizers`).
- Ensure output parity with the original Python implementation by comparing intermediate layer outputs and final logits.
