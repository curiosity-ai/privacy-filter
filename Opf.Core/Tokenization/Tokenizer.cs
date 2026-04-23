using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.ML.Tokenizers;

namespace Opf.Core.Tokenization;

public class OpfTokenizer
{
    private readonly Tokenizer _tokenizer;

    public OpfTokenizer(string directoryPath)
    {
        string configJsonPath = Path.Combine(directoryPath, "config.json");
        if (!File.Exists(configJsonPath))
        {
            throw new FileNotFoundException($"Checkpoint config not found at {configJsonPath}");
        }

        string configContent = File.ReadAllText(configJsonPath);
        var configJson = JsonNode.Parse(configContent);
        string? encodingName = configJson?["encoding"]?.ToString();
        if (string.IsNullOrEmpty(encodingName))
        {
            throw new InvalidOperationException("Checkpoint config field encoding must be a non-empty string");
        }

        _tokenizer = TiktokenTokenizer.CreateForEncoding(encodingName, null, null);
        EotTokenId = _tokenizer.EncodeToIds("<|endoftext|>", considerPreTokenization: true, considerNormalization: true).FirstOrDefault();
    }

    public int[] Encode(string text)
    {
        return _tokenizer.EncodeToIds(text).ToArray();
    }

    public string Decode(int[] tokenIds)
    {
        return _tokenizer.Decode(tokenIds);
    }

    public int EotTokenId { get; }
}
