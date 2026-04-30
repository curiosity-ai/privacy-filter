using System;
using Opf.Core.MathOps;

namespace Opf.Core.Model;

public class GroupedQueryAttention
{
    private readonly float[] _wq;
    private readonly float[] _wk;
    private readonly float[] _wv;
    private readonly float[] _wo;

    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly RotaryEmbedding _rope;

    public GroupedQueryAttention(float[] wq, float[] wk, float[] wv, float[] wo, int hiddenSize, int numHeads, int numKvHeads, RotaryEmbedding rope)
    {
        _wq = wq;
        _wk = wk;
        _wv = wv;
        _wo = wo;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = (wq.Length / hiddenSize) / numHeads;
        _rope = rope;
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int seqLen)
    {
        // 1. Projections
        float[] q = new float[seqLen * _numHeads * _headDim];
        float[] k = new float[seqLen * _numKvHeads * _headDim];
        float[] v = new float[seqLen * _numKvHeads * _headDim];

        TensorOps.Matmul(input, _wq, q, seqLen, _hiddenSize, _numHeads * _headDim);
        TensorOps.Matmul(input, _wk, k, seqLen, _hiddenSize, _numKvHeads * _headDim);
        TensorOps.Matmul(input, _wv, v, seqLen, _hiddenSize, _numKvHeads * _headDim);

        // 2. RoPE
        _rope.Forward(q, k, seqLen, _numHeads, _numKvHeads);

        // 3. Scaled Dot-Product Attention
        float scale = 1.0f / (float)Math.Sqrt(_headDim);
        int numGroups = _numHeads / _numKvHeads;

        float[] attnOut = new float[seqLen * _numHeads * _headDim];

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int kvHead = h / numGroups;

                float[] scores = new float[seqLen];
                for (int t2 = 0; t2 < seqLen; t2++)
                {
                    float score = 0;
                    for (int d = 0; d < _headDim; d++)
                    {
                        score += q[t * _numHeads * _headDim + h * _headDim + d] * k[t2 * _numKvHeads * _headDim + kvHead * _headDim + d];
                    }
                    score *= scale;

                    // Causal mask (or context masking)
                    // if (t2 > t) score = float.NegativeInfinity; // Assuming bidirectional for now since it's OPF, but need config for sliding window / bidirectional

                    scores[t2] = score;
                }

                TensorOps.Softmax(scores);

                for (int d = 0; d < _headDim; d++)
                {
                    float val = 0;
                    for (int t2 = 0; t2 < seqLen; t2++)
                    {
                        val += scores[t2] * v[t2 * _numKvHeads * _headDim + kvHead * _headDim + d];
                    }
                    attnOut[t * _numHeads * _headDim + h * _headDim + d] = val;
                }
            }
        }

        // 4. Output projection
        TensorOps.Matmul(attnOut, _wo, output, seqLen, _numHeads * _headDim, _hiddenSize);
    }
}
