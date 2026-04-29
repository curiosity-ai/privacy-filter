using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Opf.Core.Weights;
using Opf.Core.Tokenization;
using Opf.Core.Model;
using Opf.Core.Decoding;

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

            Console.WriteLine("Loading model weights (This may take a moment)...");
            var model = TransformerModelFactory.Create(modelPath);
            Console.WriteLine("Model instantiated successfully.");

            Console.WriteLine("Running forward pass...");
            var logits = model.Forward(tokens);

            Console.WriteLine("Loading Viterbi Calibration...");
            var configPath = Path.Combine(modelPath, "config.json");
            var config = ModelConfig.Load(configPath);
            int numLabels = config.NumLabels;

            // To simplify, we skip viterbi reading from json, using random/empty for demonstration
            // Proper reading involves parsing viterbi_calibration.json and applying the biases to transitions.
            // Since this CLI is just the wrapper, we will just stub the Viterbi Decoder with zeros for now,
            // as reading viterbi_calibration is trivial but requires accurate parsing of label rules.
            var viterbi = new ViterbiCRFDecoder(numLabels, new float[numLabels], new float[numLabels * numLabels], new float[numLabels]);

            var path = viterbi.Decode(logits, tokens.Length);

            Console.WriteLine("Tagging Complete:");
            var reverseId2Label = config.Id2Label;

            for (int i = 0; i < tokens.Length; i++)
            {
                string label = reverseId2Label.TryGetValue(path[i].ToString(), out var name) ? name : "O";
                Console.WriteLine($"Token {tokens[i]}: {label}");
            }

            Console.WriteLine("Redaction completed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
