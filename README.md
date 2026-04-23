# OpenAI Privacy Filter (C# Port)

This project is a modern, pure C# port of the [OpenAI Privacy Filter](https://github.com/openai/privacy-filter). It provides a bidirectional token-classification model for personally identifiable information (PII) detection and masking in text, natively targeting the .NET ecosystem.

## Features
- Pure C# implementation utilizing modern .NET features (e.g., `System.Numerics.Tensors`, SIMD intrinsics).
- Drop-in replacement for the original Python tool in pure .NET environments.
- High-performance CPU inference without reliance on PyTorch.
- Automated downloading of weights from HuggingFace (`openai/privacy-filter`).

## Components
- `Opf.Core`: The main library containing tensor operations, tokenization, model definition, and sequence decoding.
- `Opf.Cli`: A console application for performing redaction and evaluation (equivalent to the Python `opf` CLI).
- `Opf.Tests`: Unit tests validating mathematical and functional parity with the original Python code.

## Getting Started
*(Instructions on building and running the CLI will go here once implemented)*
