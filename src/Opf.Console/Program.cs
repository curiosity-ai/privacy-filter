using System;
using System.Threading.Tasks;
using Opf.Core;

namespace Opf.ConsoleApp;

public class Program
{
    public static async Task Main(string[] args)
    {
        Console.WriteLine("Opf Console Application");

        try
        {
            string checkpointPath = await CheckpointDownload.EnsureDefaultCheckpointAsync();
            Console.WriteLine($"Checkpoint ready at: {checkpointPath}");

            using var checkpoint = new Checkpoint(System.IO.Path.Combine(checkpointPath, "model.safetensors"));
            Console.WriteLine("Successfully loaded safetensors checkpoint.");

            var config = new ModelConfig();
            var transformer = new Transformer(config);
            var decoder = new ViterbiCRFDecoder(config.NumLabels ?? 33);
            var tokenizer = new Tokenizer();

            var runtime = new InferenceRuntime(transformer, decoder, tokenizer);

            string sampleText = "Alice was born on 1990-01-02.";
            var result = runtime.PredictText(sampleText);

            Console.WriteLine($"\nPrediction for: '{result.Text}'");
            foreach (var span in result.Spans)
            {
                Console.WriteLine($"- [{span.StartToken}, {span.EndToken}] {span.Label}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error during initialization: {ex.Message}");
        }
    }
}
