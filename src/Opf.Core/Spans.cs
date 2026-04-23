using System;
using System.Collections.Generic;

namespace Opf.Core;

public class SpanItem
{
    public int StartToken { get; set; }
    public int EndToken { get; set; }
    public string Label { get; set; }

    public SpanItem(int start, int end, string label)
    {
        StartToken = start;
        EndToken = end;
        Label = label;
    }
}

public static class Spans
{
    public static List<SpanItem> LabelsToSpans(List<int> decodedLabels)
    {
        var spans = new List<SpanItem>();
        int currentStart = -1;
        string currentLabel = "";

        for (int i = 0; i < decodedLabels.Count; i++)
        {
            int labelIdx = decodedLabels[i];

            // Simplified mapping for the port: assuming 0 is background (O)
            // and 1+ are some BIOES tags. A complete port would use LabelSpace.
            if (labelIdx == 0)
            {
                if (currentStart != -1)
                {
                    spans.Add(new SpanItem(currentStart, i - 1, currentLabel));
                    currentStart = -1;
                    currentLabel = "";
                }
            }
            else
            {
                if (currentStart == -1)
                {
                    currentStart = i;
                    currentLabel = $"Label_{labelIdx}";
                }
                // (Inside span, handled implicitly)
            }
        }

        if (currentStart != -1)
        {
            spans.Add(new SpanItem(currentStart, decodedLabels.Count - 1, currentLabel));
        }

        return spans;
    }
}
