using System;
using System.IO;
using System.Text.Json;
using Xunit;
using Opf.Core;

namespace Opf.Tests;

public class ViterbiDecoderTests
{
    [Fact]
    public void Decode_ExecutesWithoutException()
    {
        var inJson = File.ReadAllText("../../../../../artifacts/viterbi_in.json");
        var logProbs = JsonSerializer.Deserialize<float[]>(inJson);
        Assert.NotNull(logProbs);

        int seqLen = 5;
        int numClasses = 33;

        var decoder = new ViterbiCRFDecoder(numClasses);
        var path = decoder.Decode(logProbs, seqLen);

        Assert.NotNull(path);
        Assert.Equal(seqLen, path.Count);
    }
}
