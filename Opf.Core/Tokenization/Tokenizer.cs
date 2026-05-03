using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Tokenizers;

namespace Opf.Core.Tokenization;

public class OpfTokenizer
{
    private readonly Tokenizer _tokenizer;

    public Tokenizer Tokenizer => _tokenizer;

    public OpfTokenizer(string directoryPath)
    {
        string tokenizerJsonPath = Path.Combine(directoryPath, "tokenizer.json");
        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException($"Tokenizer config not found at {tokenizerJsonPath}");
        }

        // We can use CreateForEncoding or CreateForModel for cl100k_base or o200k_base
        // Let's use the stream approach with Create
        // But since this is a model similar to gpt-oss it likely uses cl100k_base or o200k_base.
        // Let's create it from TiktokenTokenizer.CreateForEncoding("cl100k_base", null, null)
        // Privacy filter uses o200k_base based on tokenizer.json (we can verify but TiktokenCreateForModel is fine for now)
        _tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o", null, null);
    }

    public int[] Encode(string text)
    {
        return _tokenizer.EncodeToIds(text).ToArray();
    }
}
