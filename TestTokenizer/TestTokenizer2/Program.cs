using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Tokenizers;

class Program {
    static void Main(string[] args) {
        var t1 = TiktokenTokenizer.CreateForEncoding("o200k_base", null, null);
        Console.WriteLine($"EOT Token ID: {t1.EncodeToIds("<|endoftext|>", considerPreTokenization: true, considerNormalization: true).ToArray().FirstOrDefault()}");
    }
}
