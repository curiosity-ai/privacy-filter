using System;
using System.IO;
using System.Linq;
using Opf.Core.Tokenization;
using Xunit;

namespace Opf.Tests;

public class TokenizerTests : IDisposable
{
    private readonly string _tempO200kDir;
    private readonly string _tempCl100kDir;

    public TokenizerTests()
    {
        _tempO200kDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempO200kDir);
        File.WriteAllText(Path.Combine(_tempO200kDir, "config.json"), "{\"encoding\": \"o200k_base\"}");

        _tempCl100kDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempCl100kDir);
        File.WriteAllText(Path.Combine(_tempCl100kDir, "config.json"), "{\"encoding\": \"cl100k_base\"}");
    }

    [Fact]
    public void Tokenizer_O200kBase_EncodesAndDecodesCorrectly()
    {
        var tokenizer = new OpfTokenizer(_tempO200kDir);

        string text = "hello world";
        int[] tokenIds = tokenizer.Encode(text);

        // 24912, 2375 for o200k_base based on our testing
        Assert.Equal(2, tokenIds.Length);
        Assert.Equal(24912, tokenIds[0]);
        Assert.Equal(2375, tokenIds[1]);

        string decoded = tokenizer.Decode(tokenIds);
        Assert.Equal(text, decoded);

        Assert.Equal(199999, tokenizer.EotTokenId);
    }

    [Fact]
    public void Tokenizer_Cl100kBase_EncodesAndDecodesCorrectly()
    {
        var tokenizer = new OpfTokenizer(_tempCl100kDir);

        string text = "hello world";
        int[] tokenIds = tokenizer.Encode(text);

        // 15339, 1917 for cl100k_base based on our testing
        Assert.Equal(2, tokenIds.Length);
        Assert.Equal(15339, tokenIds[0]);
        Assert.Equal(1917, tokenIds[1]);

        string decoded = tokenizer.Decode(tokenIds);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Tokenizer_ThrowsIfConfigMissing()
    {
        var emptyDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(emptyDir);

        Assert.Throws<FileNotFoundException>(() => new OpfTokenizer(emptyDir));

        Directory.Delete(emptyDir, true);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempO200kDir)) Directory.Delete(_tempO200kDir, true);
        if (Directory.Exists(_tempCl100kDir)) Directory.Delete(_tempCl100kDir, true);
    }
}
