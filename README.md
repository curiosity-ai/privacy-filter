# OpenAI Privacy Filter - C# Port

This is a pure, modern C# port of the OpenAI Privacy Filter (OPF) model. It aims to provide high-performance, local-only PII detection in .NET applications without external dependencies like Python or PyTorch.

It utilizes modern .NET features such as `System.Numerics.Tensors`, `Span<T>`, and SIMD intrinsics to achieve fast inference speeds.

## Structure
- `OpfPort.Core`: Core library containing the Transformer implementation, tensor operations, and decoding logic.
- `OpfPort.Console`: Console application for running inference.
- `OpfPort.Tests`: Unit tests and parity checks against the reference implementation.
