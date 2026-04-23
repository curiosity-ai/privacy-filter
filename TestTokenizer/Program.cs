using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Tokenizers;

class Program {
    static void Main(string[] args) {
        var t1 = TiktokenTokenizer.CreateForEncoding("o200k_base", null, null);
        Console.WriteLine($"Encode: {string.Join(",", t1.EncodeToIds("<|endoftext|>").ToArray())}");
        Console.WriteLine($"Encode allowed special: {string.Join(",", t1.EncodeToIds("<|endoftext|>", considerPreTokenization: true).ToArray())}");
    }
}
