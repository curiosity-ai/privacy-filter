using System;
using Opf.Core.MathOps;

namespace Opf.Core.Model;

public class RMSNormLayer
{
    private readonly float[] _weight;
    private readonly float _eps;

    public RMSNormLayer(float[] weight, float eps = 1e-5f)
    {
        _weight = weight;
        _eps = eps;
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output)
    {
        TensorOps.RMSNorm(input, output, _weight, _eps);
    }
}
