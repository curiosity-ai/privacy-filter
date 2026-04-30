using System;
using Opf.Core.MathOps;

namespace Opf.Core.Model;

public class TransformerBlock
{
    private readonly RMSNormLayer _attnNorm;
    private readonly GroupedQueryAttention _attn;
    private readonly RMSNormLayer _ffnNorm;
    private readonly SparseMoE _ffn;

    public TransformerBlock(RMSNormLayer attnNorm, GroupedQueryAttention attn, RMSNormLayer ffnNorm, SparseMoE ffn)
    {
        _attnNorm = attnNorm;
        _attn = attn;
        _ffnNorm = ffnNorm;
        _ffn = ffn;
    }

    public void Forward(Span<float> hiddenStates, int seqLen)
    {
        int hiddenSize = hiddenStates.Length / seqLen;

        // 1. Attention + Residual
        float[] normHiddenStates = new float[hiddenStates.Length];
        for (int t = 0; t < seqLen; t++)
        {
            var inSpan = hiddenStates.Slice(t * hiddenSize, hiddenSize);
            var outSpan = normHiddenStates.AsSpan(t * hiddenSize, hiddenSize);
            _attnNorm.Forward(inSpan, outSpan);
        }

        float[] attnOut = new float[hiddenStates.Length];
        _attn.Forward(normHiddenStates, attnOut, seqLen);

        TensorOps.Add(hiddenStates, attnOut, hiddenStates);

        // 2. FFN (MoE) + Residual
        for (int t = 0; t < seqLen; t++)
        {
            var inSpan = hiddenStates.Slice(t * hiddenSize, hiddenSize);
            var outSpan = normHiddenStates.AsSpan(t * hiddenSize, hiddenSize);
            _ffnNorm.Forward(inSpan, outSpan);
        }

        float[] ffnOut = new float[hiddenStates.Length];
        _ffn.Forward(normHiddenStates, ffnOut, seqLen);

        TensorOps.Add(hiddenStates, ffnOut, hiddenStates);
    }
}
