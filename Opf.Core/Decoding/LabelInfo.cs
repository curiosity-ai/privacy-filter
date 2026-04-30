using System;
using System.Collections.Generic;
using System.Linq;

namespace Opf.Core.Decoding;

public class LabelInfo
{
    public const string BackgroundClassLabel = "O";
    public static readonly string[] BoundaryPrefixes = { "B", "I", "E", "S" };

    public Dictionary<string, Dictionary<string, int>> BoundaryLabelLookup { get; }
    public Dictionary<int, int> TokenToSpanLabel { get; }
    public Dictionary<int, string?> TokenBoundaryTags { get; }
    public string[] SpanClassNames { get; }
    public Dictionary<string, int> SpanLabelLookup { get; }
    public int BackgroundTokenLabel { get; }
    public int BackgroundSpanLabel { get; }

    private LabelInfo(
        Dictionary<string, Dictionary<string, int>> boundaryLabelLookup,
        Dictionary<int, int> tokenToSpanLabel,
        Dictionary<int, string?> tokenBoundaryTags,
        string[] spanClassNames,
        Dictionary<string, int> spanLabelLookup,
        int backgroundTokenLabel,
        int backgroundSpanLabel)
    {
        BoundaryLabelLookup = boundaryLabelLookup;
        TokenToSpanLabel = tokenToSpanLabel;
        TokenBoundaryTags = tokenBoundaryTags;
        SpanClassNames = spanClassNames;
        SpanLabelLookup = spanLabelLookup;
        BackgroundTokenLabel = backgroundTokenLabel;
        BackgroundSpanLabel = backgroundSpanLabel;
    }

    public static LabelInfo Build(Dictionary<string, string> id2label)
    {
        var maxId = id2label.Keys.Select(int.Parse).Max();
        var classNames = new string[maxId + 1];
        foreach (var kvp in id2label)
        {
            classNames[int.Parse(kvp.Key)] = kvp.Value;
        }

        var spanClassNames = new List<string> { BackgroundClassLabel };
        var spanLabelLookup = new Dictionary<string, int> { { BackgroundClassLabel, 0 } };
        var boundaryLabelLookup = new Dictionary<string, Dictionary<string, int>>();
        var tokenToSpanLabel = new Dictionary<int, int>();
        var tokenBoundaryTags = new Dictionary<int, string?>();
        int? backgroundIdx = null;

        for (int idx = 0; idx < classNames.Length; idx++)
        {
            var name = classNames[idx];
            if (name == null) continue;

            if (name == BackgroundClassLabel)
            {
                backgroundIdx = idx;
                tokenToSpanLabel[idx] = spanLabelLookup[BackgroundClassLabel];
                tokenBoundaryTags[idx] = null;
                continue;
            }

            var parts = name.Split('-', 2);
            if (parts.Length != 2)
            {
                throw new InvalidOperationException($"Invalid label format: {name}");
            }

            var boundary = parts[0];
            var baseLabel = parts[1];

            if (!spanLabelLookup.TryGetValue(baseLabel, out int spanIdx))
            {
                spanIdx = spanClassNames.Count;
                spanClassNames.Add(baseLabel);
                spanLabelLookup[baseLabel] = spanIdx;
            }

            tokenToSpanLabel[idx] = spanIdx;
            tokenBoundaryTags[idx] = boundary;

            if (!boundaryLabelLookup.TryGetValue(baseLabel, out var mapping))
            {
                mapping = new Dictionary<string, int>();
                boundaryLabelLookup[baseLabel] = mapping;
            }
            mapping[boundary] = idx;
        }

        if (!backgroundIdx.HasValue)
        {
            throw new InvalidOperationException("Class names must include background label 'O'");
        }

        foreach (var kvp in boundaryLabelLookup)
        {
            var baseLabel = kvp.Key;
            var mapping = kvp.Value;
            var missing = BoundaryPrefixes.Except(mapping.Keys).ToList();
            if (missing.Any())
            {
                throw new InvalidOperationException($"Missing boundary classes [{string.Join(", ", missing)}] for base label {baseLabel}");
            }
        }

        return new LabelInfo(
            boundaryLabelLookup,
            tokenToSpanLabel,
            tokenBoundaryTags,
            spanClassNames.ToArray(),
            spanLabelLookup,
            backgroundIdx.Value,
            spanLabelLookup[BackgroundClassLabel]
        );
    }
}
