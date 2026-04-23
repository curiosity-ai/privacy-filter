using System;
using System.Threading.Tasks;
using Opf.Core.Weights;
using Opf.Core.Tokenization;

namespace Opf.Cli;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("Opf.Cli - C# Port of OpenAI Privacy Filter");

        string text = args.Length > 0 ? string.Join(" ", args) : "Alice was born on 1990-01-02.";

        Console.WriteLine($"Input text: {text}");

        try
        {
            var downloader = new HuggingFaceDownloader();
            string modelPath = await downloader.EnsureModelDownloadedAsync();
            Console.WriteLine($"Model loaded at: {modelPath}");

            var tokenizer = new OpfTokenizer(modelPath);
            var tokens = tokenizer.Encode(text);

            Console.WriteLine($"Tokens encoded: {tokens.Length}");
            // Skipping model execution as full architecture port with weights mapping requires more deep implementation of MXFP4,
            // but the pipeline is structured and connected.

            Console.WriteLine("Redaction completed (Stub).");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
