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

            Console.WriteLine("Loading model weights...");
            var (model, decoder) = Opf.Core.Model.ModelFactory.LoadFromDirectory(modelPath);

            Console.WriteLine("Running inference...");
            float[] logits = model.Forward(tokens);

            Console.WriteLine("Decoding sequence labels...");
            int[] labels = decoder.Decode(logits, tokens.Length);

            Console.WriteLine("Results:");
            for (int i = 0; i < tokens.Length; i++)
            {
                // Just output the token ID and the predicted label class for now
                Console.WriteLine($"Token: {tokens[i]}, Label: {labels[i]}");
            }

            Console.WriteLine("Redaction completed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
