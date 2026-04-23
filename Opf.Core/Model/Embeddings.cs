using System;

namespace Opf.Core.Model;

public class EmbeddingsLayer
{
    private readonly float[] _weight;
    private readonly int _vocabSize;
    private readonly int _hiddenSize;

    public EmbeddingsLayer(float[] weight, int vocabSize, int hiddenSize)
    {
        _weight = weight;
        _vocabSize = vocabSize;
        _hiddenSize = hiddenSize;
    }

    public void Forward(ReadOnlySpan<int> inputIds, Span<float> output)
    {
        int seqLen = inputIds.Length;
        for (int i = 0; i < seqLen; i++)
        {
            int token = inputIds[i];
            // Copy embedding vector to output
            ReadOnlySpan<float> emb = _weight.AsSpan(token * _hiddenSize, _hiddenSize);
            emb.CopyTo(output.Slice(i * _hiddenSize, _hiddenSize));
        }
    }
}
