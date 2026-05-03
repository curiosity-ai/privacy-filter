using System;
using System.Linq;
using Xunit;
using Opf.Core.Tokenization;
using Microsoft.ML.Tokenizers;
using System.Collections.Generic;

namespace Opf.Tests;

public class TokenizerTests
{
    [Fact]
    public void TestMethods()
    {
        var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o", null, null);
        var text = "Alice was born on 1990-01-02.";
        var tokenIds = tokenizer.EncodeToIds(text);

        List<string> decodedTokens = new List<string>();
        foreach (var id in tokenIds) {
            decodedTokens.Add(tokenizer.Decode(new[] { id }));
        }

        foreach (var tok in decodedTokens) Console.WriteLine($"Token: '{tok}'");
    }
}
