using System;
using System.Numerics.Tensors;

namespace Opf.Core.MathOps;

public static class TensorOps
{
    public static void RMSNorm(ReadOnlySpan<float> input, Span<float> output, ReadOnlySpan<float> weight, float eps)
    {
        int length = input.Length;
        float sumSquares = TensorPrimitives.SumOfSquares(input);

        float rsqrt = 1.0f / MathF.Sqrt(sumSquares / length + eps);

        TensorPrimitives.Multiply(input, rsqrt, output);
        TensorPrimitives.Multiply(output, weight, output);
    }

    public static void Softmax(Span<float> logits)
    {
        float maxVal = TensorPrimitives.Max(logits);
        TensorPrimitives.Subtract(logits, maxVal, logits);
        TensorPrimitives.Exp(logits, logits);
        float sum = TensorPrimitives.Sum(logits);
        TensorPrimitives.Multiply(logits, 1.0f / sum, logits);
    }

    public static void Matmul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            var aRow = a.Slice(i * k, k);
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int l = 0; l < k; l++)
                {
                    sum += aRow[l] * b[l * n + j];
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
        float alpha = 1.702f;
        float limit = 7.0f;
        for (int i = 0; i < length; i++)
        {
            float valX = x[i] > limit ? limit : x[i];
            float valY = y[i];
            if (valY > limit) valY = limit;
            if (valY < -limit) valY = -limit;

            float siluX = valX / (1.0f + MathF.Exp(-alpha * valX));
            output[i] = siluX * (valY + 1.0f);
        }
    }
}
