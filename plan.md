1. **Repository Restructuring and Documentation**
   - Move original Python codebase to `.reference` (Done).
   - Write `AGENTS.md`, `TODO.md`, `WIP.md`, `.gitignore`, `README.md` (Done).

2. **Project Initialization**
   - Create a .NET solution `Opf.sln`.
   - Create `Opf.Core` (Class Library), `Opf.Cli` (Console App), `Opf.Tests` (xUnit/MSTest).
   - Add necessary NuGet packages (e.g., `System.Numerics.Tensors`, `Microsoft.ML.Tokenizers`, etc.).

3. **HuggingFace Downloader & Safetensors Loader**
   - Implement `HuggingFaceDownloader` to fetch from `openai/privacy-filter`.
   - Implement Safetensors parsing (`SafetensorsLoader`) specifically for MXFP4/bfloat16 logic present in OPF.

4. **Tokenizer Implementation**
   - Adapt `tiktoken` BPE logic or use `Microsoft.ML.Tokenizers`. (Need to verify exact vocabulary and rules from python `opf/_core/tokenizer.py`).

5. **Tensor & SIMD Utilities**
   - Create custom lightweight tensor library or wrappers around `System.Numerics.Tensors` tailored for the inference pass, supporting basic matmul, softmax, swiglu.

6. **Transformer Architecture implementation**
   - Implement RoPE, GQA, MoE, RMSNorm.
   - Assemble `PrivacyFilterModel`.

7. **Viterbi Sequence Decoding**
   - Port `opf/_core/sequence_decoding.py` to C#.

8. **Testing & Parity Verification**
   - Instrument Python code in `.reference` to dump weights/activations for small inputs.
   - Write C# tests comparing outputs.

9. **CLI Implementation**
   - Implement `Opf.Cli` to consume the C# model and run redaction tasks.

10. **Pre-commit Steps**
    - Run all formatters, build check, test runs, and verify `AGENTS.md` instructions.

11. **Submit**
    - Finalize PR.
