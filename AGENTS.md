# OpenAI Privacy Filter C# Port Goal
This project aims to implement a pure, modern C# port of the OpenAI Privacy Filter (OPF), originally written in Python.

Key Objectives:
- Port the OPF model architecture to pure C#.
- Provide a full replacement for Python code.
- Implement efficient tensor operations utilizing modern .NET features:
  - System.Numerics.Tensors
  - SIMD intrinsics (Vector<T>, Vector256<T>, Vector512<T>, etc.)
- Re-implement all required associated code such as the tokenizer and HuggingFace model downloading and loading.
- Enable CPU inference, achieving competitive speed compared to the original PyTorch CPU implementation by leveraging hardware acceleration available in modern C#.
- Verify parity with the original Python implementation using test comparisons.

## Required Components
1. **Core Library:** Pure C# library (`Opf.Core`) providing:
   - Tensors/math operations
   - Tokenization
   - Checkpoint/safetensors loading
   - Model architecture (Embeddings, Transformer Blocks, MoE, classification head)
   - Sequence decoding (Viterbi span extraction)
2. **HuggingFace Downloader/Loader:** Logic to resolve checkpoints from `openai/privacy-filter`.
3. **Console Tool:** (`Opf.Cli`) Equivalent functionality to `opf` Python CLI.
4. **Unit Tests:** (`Opf.Tests`) Validates accuracy and parity against Python baseline.
