using System;

namespace Opf.Core;

public class RotaryEmbedding
{
    private readonly int _headDim;
    private readonly float _base;
    private readonly float _initialContextLength;
    private readonly float _scalingFactor;
    private readonly float _ntkAlpha;
    private readonly float _ntkBeta;

    public float[] CosCache { get; private set; }
    public float[] SinCache { get; private set; }

    public RotaryEmbedding(
        int headDim,
        float baseVal,
        int initialContextLength = 4096,
        float scalingFactor = 1.0f,
        float ntkAlpha = 1.0f,
        float ntkBeta = 32.0f)
    {
        _headDim = headDim;
        _base = baseVal;
        _initialContextLength = initialContextLength;
        _scalingFactor = scalingFactor;
        _ntkAlpha = ntkAlpha;
        _ntkBeta = ntkBeta;

        int maxPositions = (int)(initialContextLength * scalingFactor);
        maxPositions = Math.Max(maxPositions, initialContextLength);

        ComputeCosSin(maxPositions, out var cos, out var sin);
        CosCache = cos;
        SinCache = sin;
    }

    private void ComputeConcentrationAndInvFreq(out float concentration, out float[] invFreq)
    {
        int dHalf = _headDim / 2;
        invFreq = new float[dHalf];

        for (int i = 0; i < dHalf; i++)
        {
            invFreq[i] = (float)Math.Pow(_base, (2.0 * i) / _headDim);
        }

        if (_scalingFactor > 1.0f)
        {
            concentration = 0.1f * (float)Math.Log(_scalingFactor) + 1.0f;

            float low = (float)(dHalf * Math.Log(_initialContextLength / (_ntkBeta * 2 * Math.PI)) / Math.Log(_base));
            float high = (float)(dHalf * Math.Log(_initialContextLength / (_ntkAlpha * 2 * Math.PI)) / Math.Log(_base));

            for (int i = 0; i < dHalf; i++)
            {
                float interpolation = 1.0f / (_scalingFactor * invFreq[i]);
                float extrapolation = 1.0f / invFreq[i];

                float ramp = (i - low) / (high - low);
                float mask = 1.0f - Math.Clamp(ramp, 0.0f, 1.0f);

                invFreq[i] = interpolation * (1.0f - mask) + extrapolation * mask;
            }
        }
        else
        {
            concentration = 1.0f;
            for (int i = 0; i < dHalf; i++)
            {
                invFreq[i] = 1.0f / invFreq[i];
            }
        }
    }

    private void ComputeCosSin(int numTokens, out float[] cos, out float[] sin)
    {
        ComputeConcentrationAndInvFreq(out float concentration, out float[] invFreq);
        int dHalf = _headDim / 2;
        cos = new float[numTokens * dHalf];
        sin = new float[numTokens * dHalf];

        for (int t = 0; t < numTokens; t++)
        {
            for (int i = 0; i < dHalf; i++)
            {
                float freq = t * invFreq[i];
                cos[t * dHalf + i] = (float)Math.Cos(freq) * concentration;
                sin[t * dHalf + i] = (float)Math.Sin(freq) * concentration;
            }
        }
    }

    public void Apply(Span<float> query, Span<float> key, int batchSize, int numTokens, int numQueryHeads, int numKeyHeads)
    {
        if (numTokens > CosCache.Length / (_headDim / 2))
        {
            ComputeCosSin(numTokens, out var cos, out var sin);
            CosCache = cos;
            SinCache = sin;
        }

        int dHalf = _headDim / 2;

        ApplyRotary(query, batchSize, numTokens, numQueryHeads, dHalf);
        ApplyRotary(key, batchSize, numTokens, numKeyHeads, dHalf);
    }

    private void ApplyRotary(Span<float> tensor, int batchSize, int numTokens, int numHeads, int dHalf)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < numTokens; t++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int offset = (b * numTokens * numHeads * _headDim) + (t * numHeads * _headDim) + (h * _headDim);

                    for (int i = 0; i < dHalf; i++)
                    {
                        float x1 = tensor[offset + i];
                        float x2 = tensor[offset + i + dHalf];

                        float c = CosCache[t * dHalf + i];
                        float s = SinCache[t * dHalf + i];

                        tensor[offset + i] = x1 * c - x2 * s;
                        tensor[offset + i + dHalf] = x1 * s + x2 * c;
                    }
                }
            }
        }
    }
}
