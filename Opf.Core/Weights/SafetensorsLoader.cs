using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Opf.Core.Weights;

public class SafetensorMetadata
{
    public string Dtype { get; set; } = string.Empty;
    public long[] Shape { get; set; } = Array.Empty<long>();
    public long[] Data_offsets { get; set; } = Array.Empty<long>();
}

public class SafetensorsLoader
{
    private readonly Dictionary<string, (string FilePath, SafetensorMetadata Meta, long HeaderLength)> _tensors = new();

    public SafetensorsLoader(string directoryPath)
    {
        var files = Directory.GetFiles(directoryPath, "*.safetensors");
        foreach (var file in files)
        {
            ParseIndex(file);
        }
    }

    private void ParseIndex(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);

        long headerLength = reader.ReadInt64();
        var headerBytes = reader.ReadBytes((int)headerLength);

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var headerDict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerBytes, options);

        if (headerDict != null)
        {
            foreach (var kvp in headerDict)
            {
                if (kvp.Key == "__metadata__") continue;

                var meta = kvp.Value.Deserialize<SafetensorMetadata>(options);
                if (meta != null)
                {
                    _tensors[kvp.Key] = (filePath, meta, headerLength);
                }
            }
        }
    }

    public bool HasTensor(string name) => _tensors.ContainsKey(name);

    public (string FilePath, SafetensorMetadata Meta, long HeaderLength) GetTensorMetadata(string name)
    {
        if (_tensors.TryGetValue(name, out var data))
        {
            return data;
        }
        throw new KeyNotFoundException($"Tensor {name} not found.");
    }

    public byte[] GetTensorBytes(string name)
    {
        var data = GetTensorMetadata(name);

        long startOffset = 8 + data.HeaderLength + data.Meta.Data_offsets[0];
        long length = data.Meta.Data_offsets[1] - data.Meta.Data_offsets[0];

        byte[] buffer = new byte[(int)length];

        using var fs = new FileStream(data.FilePath, FileMode.Open, FileAccess.Read);
        fs.Seek(startOffset, SeekOrigin.Begin);
        int bytesRead = fs.Read(buffer, 0, (int)length);

        if (bytesRead != length)
        {
            throw new IOException($"Expected {length} bytes but read {bytesRead} for tensor {name}");
        }

        return buffer;
    }
}
