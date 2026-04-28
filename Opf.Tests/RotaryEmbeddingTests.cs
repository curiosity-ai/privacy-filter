using System;
using Opf.Core.Model;
using Xunit;

namespace Opf.Tests;

public class RotaryEmbeddingTests
{
    [Fact]
    public void TestRotaryEmbedding()
    {
        int headDim = 64;
        int seqLen = 10;
        int numHeads = 4;
        int numKvHeads = 2;

        var rope = new RotaryEmbedding(headDim);

        float[] query = new float[seqLen * numHeads * headDim];
        float[] key = new float[seqLen * numKvHeads * headDim];

        // Fill with ones for deterministic testing
        Array.Fill(query, 1.0f);
        Array.Fill(key, 1.0f);

        rope.Forward(query, key, seqLen, numHeads, numKvHeads);

        // Simple check to ensure we modified the array and no exceptions thrown
        // At position 0, t=0, freq=0, cos(0)=1, sin(0)=0
        // new_x1 = 1 * 1 - 1 * 0 = 1
        // new_x2 = 1 * 1 + 1 * 0 = 1
        // However at t=1, cos and sin will be different.

        // Let's check a non-zero time step (t=1, h=0, i=0)
        // headOffset = (1 * 4 + 0) * 64 = 256
        Assert.NotEqual(1.0f, query[256]);
    }
}
