using System;
using System.Buffers;
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
        var pool = ArrayPool<float>.Shared;

        // 1. Attention + Residual
        float[] normHiddenStates = pool.Rent(hiddenStates.Length);
        try
        {
            for (int t = 0; t < seqLen; t++)
            {
                var inSpan = hiddenStates.Slice(t * hiddenSize, hiddenSize);
                var outSpan = normHiddenStates.AsSpan(t * hiddenSize, hiddenSize);
                _attnNorm.Forward(inSpan, outSpan);
            }

            float[] attnOut = pool.Rent(hiddenStates.Length);
            try
            {
                _attn.Forward(normHiddenStates.AsSpan(0, hiddenStates.Length), attnOut.AsSpan(0, hiddenStates.Length), seqLen);
                TensorOps.Add(hiddenStates, attnOut.AsSpan(0, hiddenStates.Length), hiddenStates);
            }
            finally
            {
                pool.Return(attnOut);
            }

            // 2. FFN (MoE) + Residual
            for (int t = 0; t < seqLen; t++)
            {
                var inSpan = hiddenStates.Slice(t * hiddenSize, hiddenSize);
                var outSpan = normHiddenStates.AsSpan(t * hiddenSize, hiddenSize);
                _ffnNorm.Forward(inSpan, outSpan);
            }

            float[] ffnOut = pool.Rent(hiddenStates.Length);
            try
            {
                _ffn.Forward(normHiddenStates.AsSpan(0, hiddenStates.Length), ffnOut.AsSpan(0, hiddenStates.Length), seqLen);
                TensorOps.Add(hiddenStates, ffnOut.AsSpan(0, hiddenStates.Length), hiddenStates);
            }
            finally
            {
                pool.Return(ffnOut);
            }
        }
        finally
        {
            pool.Return(normHiddenStates);
        }
    }
}
