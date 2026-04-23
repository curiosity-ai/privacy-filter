using System;
using System.IO;
using System.Text.Json;
using Xunit;
using Opf.Core;

namespace Opf.Tests;

public class RMSNormTests
{
    [Fact]
    public void Forward_MatchesPyTorchOutput()
    {
        // 1. Load artifacts
        var inJson = File.ReadAllText("../../../../../artifacts/rmsnorm_in.json");
        var outJson = File.ReadAllText("../../../../../artifacts/rmsnorm_out.json");

        var inputData = JsonSerializer.Deserialize<float[]>(inJson);
        var expectedOutData = JsonSerializer.Deserialize<float[]>(outJson);

        Assert.NotNull(inputData);
        Assert.NotNull(expectedOutData);

        // Batch=2, Seq=3, Features=4 => total 24 elements
        int features = 4;
        var norm = new RMSNorm(features, eps: 1e-5f);

        // Mock scale weight
        norm.Scale[0] = 1.0f;
        norm.Scale[1] = 1.5f;
        norm.Scale[2] = 2.0f;
        norm.Scale[3] = 0.5f;

        float[] actualOutData = new float[inputData.Length];

        // 2. Run Forward
        norm.Forward(inputData, actualOutData);

        // 3. Assert
        for (int i = 0; i < inputData.Length; i++)
        {
            Assert.True(Math.Abs(expectedOutData[i] - actualOutData[i]) < 1e-5,
                $"Mismatch at index {i}: Expected {expectedOutData[i]}, Got {actualOutData[i]}");
        }
    }
}
