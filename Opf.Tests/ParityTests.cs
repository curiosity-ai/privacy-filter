using System;
using Xunit;
using Opf.Core.MathOps;
using Opf.Core.Decoding;

namespace Opf.Tests;

public class ParityTests
{
    [Fact]
    public void Softmax_WorksCorrectly()
    {
        float[] logits = { 1.0f, 2.0f, 3.0f };
        TensorOps.Softmax(logits);

        // Ensure values sum to ~1
        float sum = 0;
        foreach (var val in logits) sum += val;

        Assert.True(Math.Abs(sum - 1.0f) < 1e-4);
        // Largest input -> largest output
        Assert.True(logits[2] > logits[1] && logits[1] > logits[0]);
    }

    [Fact]
    public void ViterbiDecoder_BacktracksCorrectly()
    {
        var decoder = new ViterbiCRFDecoder(
            numLabels: 2,
            startTransitions: new[] { 0.0f, -100.0f }, // force start at 0
            transitions: new[] { 0.0f, -100.0f, -100.0f, 0.0f }, // force self-loops
            endTransitions: new[] { 0.0f, 0.0f }
        );

        float[] logits = {
            1.0f, 0.0f, // t=0
            1.0f, 0.0f  // t=1
        };

        var path = decoder.Decode(logits, 2);

        Assert.Equal(2, path.Length);
        Assert.Equal(0, path[0]);
        Assert.Equal(0, path[1]);
    }
}
