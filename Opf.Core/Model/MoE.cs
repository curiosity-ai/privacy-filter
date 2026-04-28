using System;
using System.Linq;
using Opf.Core.MathOps;

namespace Opf.Core.Model;

public class SparseMoE
{
    private readonly float[] _gateWeight;
    private readonly float[][] _expertW1;
    private readonly float[][] _expertW2;
    private readonly float[][] _expertW3; // For SwiGLU: w1 and w3 are combined in the python code into "mlp1" and "mlp2"
    private readonly int _numExperts;
    private readonly int _topK;
    private readonly int _hiddenSize;
    private readonly int _intermediateSize;

    public SparseMoE(float[] gateWeight, float[][] expertW1, float[][] expertW2, float[][] expertW3, int numExperts, int topK, int hiddenSize, int intermediateSize)
    {
        _gateWeight = gateWeight;
        _expertW1 = expertW1;
        _expertW2 = expertW2;
        _expertW3 = expertW3;
        _numExperts = numExperts;
        _topK = topK;
        _hiddenSize = hiddenSize;
        _intermediateSize = intermediateSize;
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int seqLen)
    {
        // Clear output
        output.Clear();

        // 1. Compute routing logits (input * gateWeight)
        float[] routingLogits = new float[seqLen * _numExperts];
        TensorOps.Matmul(input, _gateWeight, routingLogits, seqLen, _hiddenSize, _numExperts);

        // Process each token
        for (int t = 0; t < seqLen; t++)
        {
            // Get routing logits for current token
            ReadOnlySpan<float> tokenLogits = routingLogits.AsSpan(t * _numExperts, _numExperts);

            // 2. Select topK experts (without allocations)
            float[] topKLogits = new float[_topK];
            int[] expertIndices = new int[_topK];

            for (int k = 0; k < _topK; k++)
            {
                topKLogits[k] = float.NegativeInfinity;
                expertIndices[k] = -1;
            }

            for (int i = 0; i < _numExperts; i++)
            {
                float val = tokenLogits[i];
                // Insert into topK array
                for (int k = 0; k < _topK; k++)
                {
                    if (val > topKLogits[k])
                    {
                        // Shift down
                        for (int j = _topK - 1; j > k; j--)
                        {
                            topKLogits[j] = topKLogits[j - 1];
                            expertIndices[j] = expertIndices[j - 1];
                        }
                        topKLogits[k] = val;
                        expertIndices[k] = i;
                        break;
                    }
                }
            }

            // Softmax over topK
            TensorOps.Softmax(topKLogits);

            // 3. For each token, run selected experts (SwiGLU) and accumulate
            for (int k = 0; k < _topK; k++)
            {
                int expertIdx = expertIndices[k];
                float weight = topKLogits[k];

                ReadOnlySpan<float> tokenInput = input.Slice(t * _hiddenSize, _hiddenSize);

                // MLP1 (W1, W3)
                float[] hW1 = new float[_intermediateSize];
                float[] hW3 = new float[_intermediateSize];

                TensorOps.Matmul(tokenInput, _expertW1[expertIdx], hW1, 1, _hiddenSize, _intermediateSize);
                TensorOps.Matmul(tokenInput, _expertW3[expertIdx], hW3, 1, _hiddenSize, _intermediateSize);

                // SwiGLU
                float[] hSwiglu = new float[_intermediateSize];
                TensorOps.SwiGLU(hW1, hW3, hSwiglu);

                // MLP2 (W2)
                float[] expertOut = new float[_hiddenSize];
                TensorOps.Matmul(hSwiglu, _expertW2[expertIdx], expertOut, 1, _intermediateSize, _hiddenSize);

                // Accumulate weighted output
                for (int d = 0; d < _hiddenSize; d++)
                {
                    output[t * _hiddenSize + d] += expertOut[d] * weight;
                }
            }
        }
    }
}
