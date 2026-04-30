using System;
using System.Buffers;
using System.Collections.Generic;

namespace Opf.Core.Model;

public class TransformerModel
{
    private readonly EmbeddingsLayer _embeddings;
    private readonly List<TransformerBlock> _blocks;
    private readonly RMSNormLayer _finalNorm;
    private readonly float[] _classifierWeight;
    private readonly int _hiddenSize;
    private readonly int _numClasses;

    public TransformerModel(EmbeddingsLayer embeddings, List<TransformerBlock> blocks, RMSNormLayer finalNorm, float[] classifierWeight, int hiddenSize, int numClasses)
    {
        _embeddings = embeddings;
        _blocks = blocks;
        _finalNorm = finalNorm;
        _classifierWeight = classifierWeight;
        _hiddenSize = hiddenSize;
        _numClasses = numClasses;
    }

    public float[] Forward(int[] inputIds)
    {
        int seqLen = inputIds.Length;
        var pool = ArrayPool<float>.Shared;

        float[] hiddenStates = pool.Rent(seqLen * _hiddenSize);
        try
        {
            _embeddings.Forward(inputIds, hiddenStates.AsSpan(0, seqLen * _hiddenSize));

            foreach (var block in _blocks)
            {
                block.Forward(hiddenStates.AsSpan(0, seqLen * _hiddenSize), seqLen);
            }

            float[] normHiddenStates = pool.Rent(seqLen * _hiddenSize);
            try
            {
                for (int t = 0; t < seqLen; t++)
                {
                    var inSpan = hiddenStates.AsSpan(t * _hiddenSize, _hiddenSize);
                    var outSpan = normHiddenStates.AsSpan(t * _hiddenSize, _hiddenSize);
                    _finalNorm.Forward(inSpan, outSpan);
                }

                // Classifier projection
                float[] logits = new float[seqLen * _numClasses]; // Returned to caller
                // Full port requires Matmul here
                MathOps.TensorOps.Matmul(normHiddenStates.AsSpan(0, seqLen * _hiddenSize), _classifierWeight, logits, seqLen, _hiddenSize, _numClasses);

                return logits;
            }
            finally
            {
                pool.Return(normHiddenStates);
            }
        }
        finally
        {
            pool.Return(hiddenStates);
        }
    }
}
