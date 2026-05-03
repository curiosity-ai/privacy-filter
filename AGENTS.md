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

## Testing Parity
To generate PyTorch artifact data for unit tests, ensure you have Python 3 and `torch` installed, then run:
```bash
python3 Opf.Tests/generate_artifacts.py
```
This generates `.json` files in the `Opf.Tests` directory.
Then, you can run the parity tests using:
```bash
dotnet test Opf.Tests
```

## Hints for Future Steps
- Safetensors binary parsing, Unquantizer logic, Viterbi calibration parsing, and end-to-end logic flow is now integrated and producing coherent tagging.
- Buffer allocations inside the model hot paths have been refactored to use `ArrayPool<float>.Shared` to minimize garbage collection allocations. Further performance improvements could be made by passing continuous buffers and reducing overall loop iterations when slicing arrays, and refactoring to span bounds for output string slicing logic.
