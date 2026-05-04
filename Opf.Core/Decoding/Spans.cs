using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Tokenizers;

namespace Opf.Core.Decoding;

public static class Spans
{
    public static (List<int> CharStarts, List<int> CharEnds) TokenCharRangesForText(IReadOnlyList<int> tokenIds, Tokenizer tokenizer, string text)
    {
        var charStarts = new List<int>();
        var charEnds = new List<int>();

        var tokenBytes = new List<byte[]>();
        foreach (var tokenId in tokenIds)
        {
            var decoded = tokenizer.Decode(new[] { tokenId });
            tokenBytes.Add(Encoding.UTF8.GetBytes(decoded));
        }

        var charByteStarts = new List<int>();
        var charByteEnds = new List<int>();
        int byteCursor = 0;

        // Correctly handle UTF-16 surrogate pairs
        for (int i = 0; i < text.Length; i++)
        {
            charByteStarts.Add(byteCursor);

            string s;
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                s = text.Substring(i, 2);
                // The byte count for the combined string, we also add an entry for the second char of the pair
                int len = Encoding.UTF8.GetByteCount(s);
                byteCursor += len;
                charByteEnds.Add(byteCursor);

                i++;
                charByteStarts.Add(charByteStarts[^1]); // Point low surrogate to same start
                charByteEnds.Add(byteCursor);
            }
            else
            {
                s = text[i].ToString();
                int len = Encoding.UTF8.GetByteCount(s);
                byteCursor += len;
                charByteEnds.Add(byteCursor);
            }
        }

        int tokenByteCursor = 0;
        foreach (var rawBytes in tokenBytes)
        {
            int tokenByteStart = tokenByteCursor;
            int tokenByteEnd = tokenByteStart + rawBytes.Length;
            tokenByteCursor = tokenByteEnd;

            int startIdx = BisectRight(charByteEnds, tokenByteStart);
            int endIdx = BisectLeft(charByteStarts, tokenByteEnd);

            if (endIdx < startIdx)
            {
                endIdx = startIdx;
            }

            charStarts.Add(startIdx);
            charEnds.Add(endIdx);
        }

        return (charStarts, charEnds);
    }

    public static List<(int LabelIdx, int Start, int End)> LabelsToSpans(Dictionary<int, int> labelsByIndex, LabelInfo labelInfo)
    {
        var spans = new List<(int LabelIdx, int Start, int End)>();
        int? currentLabel = null;
        int? startIdx = null;
        int? previousIdx = null;
        int backgroundSpanLabel = labelInfo.BackgroundSpanLabel;

        var sortedKeys = labelsByIndex.Keys.ToList();
        sortedKeys.Sort();

        foreach (var tokenIdx in sortedKeys)
        {
            int labelId = labelsByIndex[tokenIdx];
            labelInfo.TokenToSpanLabel.TryGetValue(labelId, out int spanLabel);
            labelInfo.TokenBoundaryTags.TryGetValue(labelId, out string? boundaryTag);

            if (previousIdx.HasValue && tokenIdx != previousIdx.Value + 1)
            {
                if (currentLabel.HasValue && startIdx.HasValue)
                {
                    spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                }
                currentLabel = null;
                startIdx = null;
            }

            if (!labelInfo.TokenToSpanLabel.ContainsKey(labelId))
            {
                previousIdx = tokenIdx;
                continue;
            }

            bool isBackground = spanLabel == backgroundSpanLabel;
            if (isBackground)
            {
                if (currentLabel.HasValue && startIdx.HasValue)
                {
                    spans.Add((currentLabel.Value, startIdx.Value, tokenIdx));
                }
                currentLabel = null;
                startIdx = null;
                previousIdx = tokenIdx;
                continue;
            }

            if (boundaryTag == "S")
            {
                if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
                {
                    spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                }
                spans.Add((spanLabel, tokenIdx, tokenIdx + 1));
                currentLabel = null;
                startIdx = null;
            }
            else if (boundaryTag == "B")
            {
                if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
                {
                    spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                }
                currentLabel = spanLabel;
                startIdx = tokenIdx;
            }
            else if (boundaryTag == "I")
            {
                if (!currentLabel.HasValue || currentLabel.Value != spanLabel)
                {
                    if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
                    {
                        spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                    }
                    currentLabel = spanLabel;
                    startIdx = tokenIdx;
                }
            }
            else if (boundaryTag == "E")
            {
                if (!currentLabel.HasValue || currentLabel.Value != spanLabel || !startIdx.HasValue)
                {
                    if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
                    {
                        spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                    }
                    spans.Add((spanLabel, tokenIdx, tokenIdx + 1));
                    currentLabel = null;
                    startIdx = null;
                }
                else
                {
                    spans.Add((currentLabel.Value, startIdx.Value, tokenIdx + 1));
                    currentLabel = null;
                    startIdx = null;
                }
            }
            else
            {
                if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
                {
                    spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
                }
                currentLabel = null;
                startIdx = null;
            }

            previousIdx = tokenIdx;
        }

        if (currentLabel.HasValue && startIdx.HasValue && previousIdx.HasValue)
        {
            spans.Add((currentLabel.Value, startIdx.Value, previousIdx.Value + 1));
        }

        return spans;
    }

    public static List<(int LabelIdx, int Start, int End)> TokenSpansToCharSpans(
        List<(int LabelIdx, int Start, int End)> spans,
        List<int> charStarts,
        List<int> charEnds)
    {
        var converted = new List<(int LabelIdx, int Start, int End)>();
        if (spans == null || spans.Count == 0) return converted;

        foreach (var span in spans)
        {
            if (!(0 <= span.Start && span.Start < span.End && span.End <= charStarts.Count)) continue;

            int charStart = charStarts[span.Start];
            int charEnd = charEnds[span.End - 1];

            if (charEnd <= charStart) continue;

            converted.Add((span.LabelIdx, charStart, charEnd));
        }

        return converted;
    }

    private static int BisectRight(List<int> a, int x)
    {
        int lo = 0;
        int hi = a.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (x < a[mid]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    private static int BisectLeft(List<int> a, int x)
    {
        int lo = 0;
        int hi = a.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (a[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}
