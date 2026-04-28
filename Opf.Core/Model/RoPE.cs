using System;
using System.Numerics.Tensors;

namespace Opf.Core.Model;

public class RotaryEmbedding
{
    private readonly int _headDim;
    private readonly float _baseFreq;
    private readonly int _initialContextLength;
    private readonly float _scalingFactor;
    private readonly float _ntkAlpha;
    private readonly float _ntkBeta;

    private float[] _cosCache;
    private float[] _sinCache;

    public RotaryEmbedding(
        int headDim,
        float baseFreq = 10000.0f,
        int initialContextLength = 4096,
        float scalingFactor = 1.0f,
        float ntkAlpha = 1.0f,
        float ntkBeta = 32.0f)
    {
        _headDim = headDim;
        _baseFreq = baseFreq;
        _initialContextLength = initialContextLength;
        _scalingFactor = scalingFactor;
        _ntkAlpha = ntkAlpha;
        _ntkBeta = ntkBeta;

        int maxPositions = (int)Math.Max(initialContextLength * scalingFactor, initialContextLength);
        _cosCache = new float[maxPositions * (headDim / 2)];
        _sinCache = new float[maxPositions * (headDim / 2)];

        ComputeCosSin(maxPositions);
    }

    private void ComputeCosSin(int numTokens)
    {
        float concentration = 1.0f;
        float[] invFreq = new float[_headDim / 2];

        for (int i = 0; i < _headDim; i += 2)
        {
            invFreq[i / 2] = 1.0f / (float)Math.Pow(_baseFreq, (float)i / _headDim);
        }

        if (_scalingFactor > 1.0f)
        {
            concentration = 0.1f * (float)Math.Log(_scalingFactor) + 1.0f;
            float dHalf = _headDim / 2.0f;
            float low = (float)(dHalf * Math.Log(_initialContextLength / (_ntkBeta * 2 * Math.PI)) / Math.Log(_baseFreq));
            float high = (float)(dHalf * Math.Log(_initialContextLength / (_ntkAlpha * 2 * Math.PI)) / Math.Log(_baseFreq));

            for (int i = 0; i < _headDim / 2; i++)
            {
                float freq = (float)Math.Pow(_baseFreq, (float)(i * 2) / _headDim);
                float interpolation = 1.0f / (_scalingFactor * freq);
                float extrapolation = 1.0f / freq;

                float ramp = (i - low) / (high - low);
                ramp = Math.Max(0.0f, Math.Min(1.0f, ramp));
                float mask = 1.0f - ramp;

                invFreq[i] = interpolation * (1.0f - mask) + extrapolation * mask;
            }
        }

        if (numTokens * (_headDim / 2) > _cosCache.Length)
        {
            _cosCache = new float[numTokens * (_headDim / 2)];
            _sinCache = new float[numTokens * (_headDim / 2)];
        }

        for (int t = 0; t < numTokens; t++)
        {
            for (int i = 0; i < _headDim / 2; i++)
            {
                float freq = t * invFreq[i];
                int index = t * (_headDim / 2) + i;
                _cosCache[index] = (float)Math.Cos(freq) * concentration;
                _sinCache[index] = (float)Math.Sin(freq) * concentration;
            }
        }
    }

    public void Forward(Span<float> query, Span<float> key, int seqLen, int numHeads, int numKvHeads)
    {
        if (seqLen * (_headDim / 2) > _cosCache.Length)
        {
            ComputeCosSin(seqLen);
        }

        // Apply RoPE to Query
        ApplyRotary(query, seqLen, numHeads);

        // Apply RoPE to Key
        ApplyRotary(key, seqLen, numKvHeads);
    }

    private void ApplyRotary(Span<float> x, int seqLen, int numHeads)
    {
        for (int t = 0; t < seqLen; t++)
        {
            int cosSinOffset = t * (_headDim / 2);

            for (int h = 0; h < numHeads; h++)
            {
                int headOffset = (t * numHeads + h) * _headDim;

                for (int i = 0; i < _headDim / 2; i++)
                {
                    float x1 = x[headOffset + i];
                    float x2 = x[headOffset + i + _headDim / 2];

                    float cos = _cosCache[cosSinOffset + i];
                    float sin = _sinCache[cosSinOffset + i];

                    x[headOffset + i] = x1 * cos - x2 * sin;
                    x[headOffset + i + _headDim / 2] = x2 * cos + x1 * sin;
                }
            }
        }
    }
}
