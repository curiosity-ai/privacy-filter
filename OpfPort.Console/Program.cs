using System;
using System.IO;
using System.Threading.Tasks;
using OpfPort.Core;
using System.Linq;

namespace OpfPort.ConsoleApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("OpenAI Privacy Filter - C# Port");

            string modelDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".opf", "privacy_filter");
            await ModelDownloader.DownloadModelAsync(modelDir);

            Console.WriteLine("Model downloaded. Initializing components...");

            var tokenizer = new TokenizerWrapper(Path.Combine(modelDir, "tokenizer.json"));

            string text = "Alice was born on 1990-01-02.";
            Console.WriteLine($"Input: {text}");

            var tokens = tokenizer.Encode(text);
            Console.WriteLine($"Tokens: {string.Join(", ", tokens)}");

            Console.WriteLine("Loading model...");
            var config = ConfigLoader.LoadConfig(Path.Combine(modelDir, "config.json"));
            var transformer = new Transformer(config);

            Console.WriteLine("Running inference...");
            var tokenArray = tokens.ToArray();
            var logits = transformer.Forward(tokenArray);

            Console.WriteLine("Decoding spans...");
            var decoder = new ViterbiDecoder();
            var spans = decoder.Decode(logits, tokenArray.Length);

            foreach (var span in spans)
            {
                Console.WriteLine($"Detected {span.Label} at tokens {span.Start}-{span.End}");
            }

            Console.WriteLine("Done.");
        }
    }
}
