using System;
using System.Collections.Generic;

namespace Opf.Core.Decoding;

public class ViterbiCRFDecoder
{
    private readonly int _numLabels;
    private readonly float[] _startTransitions;
    private readonly float[] _transitions;
    private readonly float[] _endTransitions;

    public ViterbiCRFDecoder(int numLabels, float[] startTransitions, float[] transitions, float[] endTransitions)
    {
        _numLabels = numLabels;
        _startTransitions = startTransitions;
        _transitions = transitions;
        _endTransitions = endTransitions;
    }

    public int[] Decode(ReadOnlySpan<float> logits, int seqLen)
    {
        // Simple Viterbi implementation
        var path = new int[seqLen * _numLabels];
        var scores = new float[_numLabels];
        var nextScores = new float[_numLabels];

        // Initialization
        for (int i = 0; i < _numLabels; i++)
        {
            scores[i] = _startTransitions[i] + logits[i];
        }

        // Recursion
        for (int t = 1; t < seqLen; t++)
        {
            int logitOffset = t * _numLabels;
            for (int i = 0; i < _numLabels; i++)
            {
                float maxScore = float.NegativeInfinity;
                int maxIdx = 0;
                for (int j = 0; j < _numLabels; j++)
                {
                    float score = scores[j] + _transitions[j * _numLabels + i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxIdx = j;
                    }
                }
                nextScores[i] = maxScore + logits[logitOffset + i];
                path[t * _numLabels + i] = maxIdx;
            }
            Array.Copy(nextScores, scores, _numLabels);
        }

        // Termination
        float bestFinalScore = float.NegativeInfinity;
        int bestLastState = 0;
        for (int i = 0; i < _numLabels; i++)
        {
            float score = scores[i] + _endTransitions[i];
            if (score > bestFinalScore)
            {
                bestFinalScore = score;
                bestLastState = i;
            }
        }

        // Backtrack
        var result = new int[seqLen];
        result[seqLen - 1] = bestLastState;
        for (int t = seqLen - 1; t > 0; t--)
        {
            result[t - 1] = path[t * _numLabels + result[t]];
        }

        return result;
    }
}
