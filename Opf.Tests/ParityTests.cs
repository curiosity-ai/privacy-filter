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

    [Fact]
    public void Mxfp4Unquantizer_WorksCorrectly()
    {
        // 0b_0001_0010 -> idx_hi = 1 (0.5f), idx_lo = 2 (1.0f)
        byte[] blocks = { 0b_0001_0010 };
        // Scale 127 -> exp = 0
        int[] scales = { 127 };

        float[] result = Opf.Core.Weights.Mxfp4Unquantizer.Unquantize(blocks, scales, rows: 1, columns: 1);

        Assert.Equal(2, result.Length);
        Assert.Equal(1.0f, result[0]); // idx_lo
        Assert.Equal(0.5f, result[1]); // idx_hi

        // Another block: 0b_0000_0100 -> idx_hi = 0 (0.0f), idx_lo = 4 (2.0f)
        // Scale 128 -> exp = 1
        blocks = new byte[] { 0b_0000_0100 };
        scales = new int[] { 128 };

        result = Opf.Core.Weights.Mxfp4Unquantizer.Unquantize(blocks, scales, rows: 1, columns: 1);

        Assert.Equal(2, result.Length);
        Assert.Equal(4.0f, result[0]); // 2.0f * 2^1 = 4.0f
        Assert.Equal(0.0f, result[1]); // 0.0f * 2^1 = 0.0f
    }
}
