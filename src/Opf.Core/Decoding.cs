using System;
using System.Collections.Generic;
using System.Linq;

namespace Opf.Core;

public class ViterbiCRFDecoder
{
    private readonly float[] _startScores;
    private readonly float[] _endScores;
    private readonly float[] _transitionScores;
    private readonly int _numClasses;

    public ViterbiCRFDecoder(int numClasses)
    {
        _numClasses = numClasses;
        _startScores = new float[numClasses];
        _endScores = new float[numClasses];
        _transitionScores = new float[numClasses * numClasses];

        // For simplicity in this port context, we initialize with zeroes/defaults
        // A complete port would map token_to_span_label and boundary tags perfectly.
        Array.Fill(_startScores, 0.0f);
        Array.Fill(_endScores, 0.0f);
        Array.Fill(_transitionScores, 0.0f);
    }

    public List<int> Decode(float[] tokenLogProbs, int seqLen)
    {
        if (seqLen == 0) return new List<int>();

        float[] scores = new float[_numClasses];
        for (int i = 0; i < _numClasses; i++)
        {
            scores[i] = tokenLogProbs[i] + _startScores[i];
        }

        int[] backpointers = new int[(seqLen - 1) * _numClasses];

        for (int idx = 1; idx < seqLen; idx++)
        {
            float[] newScores = new float[_numClasses];
            for (int nextClass = 0; nextClass < _numClasses; nextClass++)
            {
                float bestScore = float.NegativeInfinity;
                int bestPath = -1;

                for (int prevClass = 0; prevClass < _numClasses; prevClass++)
                {
                    float transScore = _transitionScores[prevClass * _numClasses + nextClass];
                    float currentScore = scores[prevClass] + transScore;

                    if (currentScore > bestScore)
                    {
                        bestScore = currentScore;
                        bestPath = prevClass;
                    }
                }

                newScores[nextClass] = bestScore + tokenLogProbs[idx * _numClasses + nextClass];
                backpointers[(idx - 1) * _numClasses + nextClass] = bestPath;
            }
            scores = newScores;
        }

        float finalBestScore = float.NegativeInfinity;
        int lastLabel = -1;
        for (int i = 0; i < _numClasses; i++)
        {
            float finalScore = scores[i] + _endScores[i];
            if (finalScore > finalBestScore)
            {
                finalBestScore = finalScore;
                lastLabel = i;
            }
        }

        if (lastLabel == -1) lastLabel = 0; // fallback

        int[] path = new int[seqLen];
        path[seqLen - 1] = lastLabel;

        for (int idx = seqLen - 2; idx >= 0; idx--)
        {
            lastLabel = backpointers[idx * _numClasses + lastLabel];
            path[idx] = lastLabel;
        }

        return path.ToList();
    }
}
