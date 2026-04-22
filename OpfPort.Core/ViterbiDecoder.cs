using System;
using System.Collections.Generic;
using System.Linq;

namespace OpfPort.Core
{
    public class SpanLabel
    {
        public string Label { get; set; } = string.Empty;
        public int Start { get; set; }
        public int End { get; set; }
    }

    public class ViterbiDecoder
    {
        // Simple mock of Viterbi decoder. Full logic depends on parsing the transitions
        // OPF uses BIOES tags.
        // We will output dummy spans for now to wire things up.

        public List<SpanLabel> Decode(float[] logits, int seqLen)
        {
            var spans = new List<SpanLabel>();

            // Just a mock decoding
            if (seqLen > 5)
            {
                spans.Add(new SpanLabel { Label = "private_person", Start = 0, End = 1 });
            }

            return spans;
        }
    }
}
