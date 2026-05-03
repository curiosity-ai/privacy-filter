using System;
using System.Collections.Generic;
using Xunit;
using Opf.Core.Decoding;
using Opf.Core.Tokenization;
using Microsoft.ML.Tokenizers;

namespace Opf.Tests;

public class SpansTests
{
    [Fact]
    public void TokenCharRangesForText_ReturnsCorrectOffsets()
    {
        var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o", null, null);
        var text = "Alice was born on 1990-01-02.";
        var tokenIds = tokenizer.EncodeToIds(text);

        var (charStarts, charEnds) = Spans.TokenCharRangesForText(tokenIds, tokenizer, text);

        Assert.Equal(tokenIds.Count, charStarts.Count);
        Assert.Equal(tokenIds.Count, charEnds.Count);

        // Assert text offsets
        for (int i = 0; i < tokenIds.Count; i++)
        {
            var tokenStr = tokenizer.Decode(new[] { tokenIds[i] });
            var substring = text.Substring(charStarts[i], charEnds[i] - charStarts[i]);
            Assert.Equal(tokenStr, substring);
        }
    }
}
