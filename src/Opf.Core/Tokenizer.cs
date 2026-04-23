using System;
using System.Collections.Generic;

namespace Opf.Core;

// Simplified tokenizer mock to fulfill basic structural requirement.
// A full Tiktoken port is highly complex and usually handled via a Nuget package like Microsoft.ML.Tokenizers
public class Tokenizer
{
    public List<int> Encode(string text)
    {
        // Mock encoding
        var tokens = new List<int>();
        foreach (char c in text)
        {
            tokens.Add((int)c);
        }
        return tokens;
    }

    public string Decode(List<int> tokens)
    {
        var chars = new char[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            chars[i] = (char)tokens[i];
        }
        return new string(chars);
    }
}
