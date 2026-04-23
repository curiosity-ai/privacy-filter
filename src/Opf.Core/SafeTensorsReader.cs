using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Opf.Core;

public class TensorInfo
{
    [JsonPropertyName("dtype")]
    public string Dtype { get; set; } = string.Empty;
    [JsonPropertyName("shape")]
    public int[] Shape { get; set; } = Array.Empty<int>();
    [JsonPropertyName("data_offsets")]
    public long[] DataOffsets { get; set; } = Array.Empty<long>();
}

public class SafeTensorsReader : IDisposable
{
    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private readonly Dictionary<string, TensorInfo> _index = new();
    private readonly long _dataOffset;

    public SafeTensorsReader(string filePath)
    {
        _stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        _reader = new BinaryReader(_stream);

        // SafeTensors format uses an 8-byte little-endian unsigned integer for the header size
        ulong headerSizeBytes = _reader.ReadUInt64();

        if (headerSizeBytes > int.MaxValue)
        {
             throw new InvalidOperationException("Header size is too large.");
        }

        byte[] headerBytes = _reader.ReadBytes((int)headerSizeBytes);
        var options = new JsonSerializerOptions { AllowTrailingCommas = true };

        using var document = JsonDocument.Parse(headerBytes);
        foreach (var prop in document.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;
            var tensorInfo = JsonSerializer.Deserialize<TensorInfo>(prop.Value.GetRawText(), options);
            if (tensorInfo != null)
            {
                _index[prop.Name] = tensorInfo;
            }
        }

        _dataOffset = 8 + (long)headerSizeBytes;
    }

    public bool HasTensor(string name) => _index.ContainsKey(name);

    public float[] GetTensor(string name)
    {
        if (!_index.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Tensor {name} not found");

        long start = info.DataOffsets[0];
        long end = info.DataOffsets[1];
        int byteLength = (int)(end - start);

        _stream.Seek(_dataOffset + start, SeekOrigin.Begin);
        byte[] data = _reader.ReadBytes(byteLength);

        // Currently supporting bf16 and fp32 (bf16 will need conversion)
        if (info.Dtype == "F32")
        {
            float[] result = new float[byteLength / 4];
            Buffer.BlockCopy(data, 0, result, 0, byteLength);
            return result;
        }
        else if (info.Dtype == "BF16")
        {
            float[] result = new float[byteLength / 2];
            for (int i = 0; i < result.Length; i++)
            {
                ushort bf16 = BitConverter.ToUInt16(data, i * 2);
                uint f32 = (uint)bf16 << 16;
                result[i] = BitConverter.Int32BitsToSingle((int)f32);
            }
            return result;
        }

        throw new NotSupportedException($"Dtype {info.Dtype} not supported");
    }

    public void Dispose()
    {
        _reader.Dispose();
        _stream.Dispose();
    }
}
