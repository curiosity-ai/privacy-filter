using System;
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
        float[] hiddenStates = new float[seqLen * _hiddenSize];

        _embeddings.Forward(inputIds, hiddenStates);

        foreach (var block in _blocks)
        {
            block.Forward(hiddenStates, seqLen);
        }

        float[] normHiddenStates = new float[seqLen * _hiddenSize];
        _finalNorm.Forward(hiddenStates, normHiddenStates);

        // Classifier projection
        float[] logits = new float[seqLen * _numClasses];
        // Full port requires Matmul here
        MathOps.TensorOps.Matmul(normHiddenStates, _classifierWeight, logits, seqLen, _hiddenSize, _numClasses);

        return logits;
    }
}
