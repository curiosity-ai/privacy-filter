using System;
using System.Reflection;
using Microsoft.ML.Tokenizers;

class Program {
    static void Main() {
        foreach (var m in typeof(Tokenizer).GetMethods(BindingFlags.Public | BindingFlags.Static)) {
            Console.WriteLine(m.Name);
        }
    }
}
