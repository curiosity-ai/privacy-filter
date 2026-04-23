using System;
using System.Numerics.Tensors;

namespace Opf.Core;

public class RMSNorm
{
    private readonly int _numFeatures;
    private readonly float _eps;
    public float[] Scale { get; }

    public RMSNorm(int numFeatures, float eps = 1e-05f)
    {
        _numFeatures = numFeatures;
        _eps = eps;
        Scale = new float[numFeatures];
        Array.Fill(Scale, 1.0f);
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length % _numFeatures != 0 || output.Length != input.Length)
        {
            throw new ArgumentException("Input and output lengths must be identical and a multiple of numFeatures.");
        }

        int numVectors = input.Length / _numFeatures;

        for (int i = 0; i < numVectors; i++)
        {
            var inSlice = input.Slice(i * _numFeatures, _numFeatures);
            var outSlice = output.Slice(i * _numFeatures, _numFeatures);

            float sumSq = TensorPrimitives.SumOfSquares(inSlice);
            float meanSq = sumSq / _numFeatures;
            float rsqrt = 1.0f / (float)Math.Sqrt(meanSq + _eps);

            TensorPrimitives.Multiply(inSlice, rsqrt, outSlice);
            TensorPrimitives.Multiply(outSlice, Scale, outSlice);
        }
    }
}
