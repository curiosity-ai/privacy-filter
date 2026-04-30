using System;

namespace Opf.Core.Decoding;

public static class ViterbiDecoderBuilder
{
    private const float NegInf = -1e9f;

    public static ViterbiCRFDecoder Build(LabelInfo labelInfo, ViterbiCalibrationBiases? biases = null)
    {
        biases ??= new ViterbiCalibrationBiases();

        int numClasses = labelInfo.TokenToSpanLabel.Count;
        var startScores = new float[numClasses];
        var endScores = new float[numClasses];
        var transitionScores = new float[numClasses * numClasses];

        Array.Fill(startScores, NegInf);
        Array.Fill(endScores, NegInf);
        Array.Fill(transitionScores, NegInf);

        int backgroundTokenIdx = labelInfo.BackgroundTokenLabel;
        int backgroundSpanIdx = labelInfo.BackgroundSpanLabel;
        var tokenBoundaryTags = labelInfo.TokenBoundaryTags;
        var tokenToSpanLabel = labelInfo.TokenToSpanLabel;

        for (int idx = 0; idx < numClasses; idx++)
        {
            tokenBoundaryTags.TryGetValue(idx, out string? tag);
            tokenToSpanLabel.TryGetValue(idx, out int spanLabel);

            if (tag == "B" || tag == "S" || idx == backgroundTokenIdx)
            {
                startScores[idx] = 0.0f;
            }

            if (tag == "E" || tag == "S" || idx == backgroundTokenIdx)
            {
                endScores[idx] = 0.0f;
            }

            for (int nextIdx = 0; nextIdx < numClasses; nextIdx++)
            {
                tokenBoundaryTags.TryGetValue(nextIdx, out string? nextTag);
                tokenToSpanLabel.TryGetValue(nextIdx, out int nextSpanLabel);

                if (IsValidTransition(
                        prevTag: tag,
                        prevSpan: spanLabel,
                        nextTag: nextTag,
                        nextSpan: nextSpanLabel,
                        backgroundTokenIdx: backgroundTokenIdx,
                        backgroundSpanIdx: backgroundSpanIdx,
                        nextIdx: nextIdx))
                {
                    transitionScores[idx * numClasses + nextIdx] = TransitionBias(
                        biases,
                        prevTag: tag,
                        prevSpan: spanLabel,
                        nextTag: nextTag,
                        nextSpan: nextSpanLabel,
                        backgroundTokenIdx: backgroundTokenIdx,
                        backgroundSpanIdx: backgroundSpanIdx,
                        prevIdx: idx,
                        nextIdx: nextIdx);
                }
            }
        }

        return new ViterbiCRFDecoder(numClasses, startScores, transitionScores, endScores);
    }

    private static bool IsValidTransition(
        string? prevTag,
        int prevSpan,
        string? nextTag,
        int nextSpan,
        int backgroundTokenIdx,
        int backgroundSpanIdx,
        int nextIdx)
    {
        bool nextIsBackground = nextSpan == backgroundSpanIdx || nextIdx == backgroundTokenIdx;
        if (nextTag == null && !nextIsBackground)
        {
            return false;
        }

        if (prevTag == null)
        {
            return nextIsBackground || nextTag == "B" || nextTag == "S";
        }

        bool prevIsBackground = prevSpan == backgroundSpanIdx;

        if (prevIsBackground)
        {
            return nextIsBackground || nextTag == "B" || nextTag == "S";
        }

        if (prevTag == "E" || prevTag == "S")
        {
            return nextIsBackground || nextTag == "B" || nextTag == "S";
        }

        if (prevTag == "B" || prevTag == "I")
        {
            bool sameSpan = prevSpan == nextSpan;
            return sameSpan && (nextTag == "I" || nextTag == "E");
        }

        return false;
    }

    private static float TransitionBias(
        ViterbiCalibrationBiases biases,
        string? prevTag,
        int prevSpan,
        string? nextTag,
        int nextSpan,
        int backgroundTokenIdx,
        int backgroundSpanIdx,
        int prevIdx,
        int nextIdx)
    {
        bool prevIsBackground = prevSpan == backgroundSpanIdx || prevIdx == backgroundTokenIdx;
        bool nextIsBackground = nextSpan == backgroundSpanIdx || nextIdx == backgroundTokenIdx;

        if (prevIsBackground)
        {
            if (nextIsBackground) return biases.TransitionBiasBackgroundStay;
            if (nextTag == "B" || nextTag == "S") return biases.TransitionBiasBackgroundToStart;
            return 0.0f;
        }

        if (prevTag == "B" || prevTag == "I")
        {
            if (nextTag == "I" && prevSpan == nextSpan) return biases.TransitionBiasInsideToContinue;
            if (nextTag == "E" && prevSpan == nextSpan) return biases.TransitionBiasInsideToEnd;
            return 0.0f;
        }

        if (prevTag == "E" || prevTag == "S")
        {
            if (nextIsBackground) return biases.TransitionBiasEndToBackground;
            if (nextTag == "B" || nextTag == "S") return biases.TransitionBiasEndToStart;
            return 0.0f;
        }

        return 0.0f;
    }
}
