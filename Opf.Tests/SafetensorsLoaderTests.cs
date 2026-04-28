using System;
using System.IO;
using System.Text;
using System.Text.Json;
using Xunit;
using Opf.Core.Weights;

namespace Opf.Tests;

public class SafetensorsLoaderTests : IDisposable
{
    private readonly string _testDirPath;

    public SafetensorsLoaderTests()
    {
        _testDirPath = Path.Combine(Path.GetTempPath(), "SafetensorsTestDir_" + Guid.NewGuid());
        Directory.CreateDirectory(_testDirPath);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDirPath))
        {
            Directory.Delete(_testDirPath, true);
        }
    }

    [Fact]
    public void SafetensorsLoader_CanReadTensorBytes()
    {
        // Setup: Create a dummy safetensors file
        string filePath = Path.Combine(_testDirPath, "dummy.safetensors");

        string headerJson = "{\"tensor_a\": {\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [0, 8]}, \"__metadata__\": {\"format\": \"pt\"}}";

        // Pad header with spaces so it matches typical safetensor alignments if needed, but not strictly necessary for this simple test.
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
        long headerLength = headerBytes.Length;

        // dummy tensor payload: 8 bytes (2 floats)
        byte[] payloadBytes = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };

        using (var fs = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(fs))
        {
            writer.Write(headerLength);
            writer.Write(headerBytes);
            writer.Write(payloadBytes);
        }

        // Test
        var loader = new SafetensorsLoader(_testDirPath);

        Assert.True(loader.HasTensor("tensor_a"));
        Assert.False(loader.HasTensor("tensor_b"));

        var metadata = loader.GetTensorMetadata("tensor_a");
        Assert.Equal("F32", metadata.Meta.Dtype);
        Assert.Equal(2, metadata.Meta.Shape[0]);

        byte[] readBytes = loader.GetTensorBytes("tensor_a");
        Assert.Equal(8, readBytes.Length);
        Assert.Equal(payloadBytes, readBytes);
    }
}
