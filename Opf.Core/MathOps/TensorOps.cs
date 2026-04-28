using System;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Opf.Core.MathOps;

public static class TensorOps
{
    public static void RMSNorm(ReadOnlySpan<float> input, Span<float> output, ReadOnlySpan<float> weight, float eps)
    {
        int length = input.Length;
        float sumSquares = 0f;

        // Compute sum of squares
        for (int i = 0; i < length; i++)
        {
            float v = input[i];
            sumSquares += v * v;
        }

        float rsqrt = 1.0f / MathF.Sqrt(sumSquares / length + eps);

        // Normalize and scale
        for (int i = 0; i < length; i++)
        {
            output[i] = input[i] * rsqrt * weight[i];
        }
    }

    public static void Softmax(Span<float> logits)
    {
        TensorPrimitives.SoftMax(logits, logits);
    }

    public static void Matmul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        // Dense matmul: C = A * B
        // A is [M, K], B is [K, N], C is [M, N]
        // NOTE: We assume 'b' is row-major. For better performance 'b' should be transposed beforehand.
        // Doing this synchronously because Span<float> cannot be captured in a lambda expression.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        TensorPrimitives.Add(a, b, c);
    }

    public static void Add(ReadOnlySpan<float> a, float b, Span<float> c)
    {
        TensorPrimitives.Add(a, b, c);
    }

    public static void Multiply(ReadOnlySpan<float> a, float b, Span<float> c)
    {
        TensorPrimitives.Multiply(a, b, c);
    }

    public static void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        TensorPrimitives.Multiply(a, b, c);
    }

    public static void SwiGLU(Span<float> x, Span<float> y, Span<float> output)
    {
        int length = x.Length;
        for (int i = 0; i < length; i++)
        {
            float valX = x[i];
            // SiLU (Swish with beta=1)
            float siluX = valX / (1.0f + MathF.Exp(-valX));
            output[i] = siluX * y[i];
        }
    }
}
