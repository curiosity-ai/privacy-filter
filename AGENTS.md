# OpenAI Privacy Filter (C# Port) - Goal and Outline

The primary goal of this repository is to provide a pure, modern C# implementation of the OpenAI Privacy Filter (OPF), originally written in Python. This implementation uses modern features like Tensors, SIMD, and vectors available in .NET 8+.

## Key Objectives

1.  **Zero Dependencies on Python**: A complete rewrite of the model inference, decoding, and tokenizer handling in C#.
2.  **High Performance**: Utilize `System.Numerics.Tensors` and `System.Runtime.Intrinsics` to accelerate tensor operations.
3.  **Accuracy**: Match the output and intermediate tensor states of the original PyTorch model exactly.
4.  **Usability**: Provide a clean API (`Opf.Core`) and an easy-to-use CLI (`Opf.Console`).

## Automated Checks
Ensure that all unit tests pass:
`dotnet test`
Ensure that the code builds without warnings or errors:
`dotnet build`
