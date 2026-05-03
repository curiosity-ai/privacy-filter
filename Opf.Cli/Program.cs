using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
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
            var labelInfo = LabelInfo.Build(config.Id2Label);

            var viterbiPath = Path.Combine(modelPath, "viterbi_calibration.json");
            var calibration = ViterbiCalibration.Load(viterbiPath);
            var biases = calibration.OperatingPoints.Default.Biases;

            var viterbi = ViterbiDecoderBuilder.Build(labelInfo, biases);

            var path = viterbi.Decode(logits, tokens.Length);

            var predictedLabelsByIndex = new Dictionary<int, int>();
            for (int i = 0; i < path.Length; i++)
            {
                predictedLabelsByIndex[i] = path[i];
            }

            var predictedTokenSpans = Spans.LabelsToSpans(predictedLabelsByIndex, labelInfo);

            var (charStarts, charEnds) = Spans.TokenCharRangesForText(tokens, tokenizer.Tokenizer, text);

            var predictedCharSpans = Spans.TokenSpansToCharSpans(predictedTokenSpans, charStarts, charEnds);

            Console.WriteLine("\nDetected Spans:");
            foreach (var span in predictedCharSpans)
            {
                string label = labelInfo.SpanClassNames.Length > span.LabelIdx ? labelInfo.SpanClassNames[span.LabelIdx] : $"label_{span.LabelIdx}";
                string spanText = text.Substring(span.Start, span.End - span.Start);
                Console.WriteLine($"- [{span.Start}, {span.End}] {label}: '{spanText}'");
            }

            string redactedText = text;
            int offset = 0;
            foreach (var span in predictedCharSpans)
            {
                string label = labelInfo.SpanClassNames.Length > span.LabelIdx ? labelInfo.SpanClassNames[span.LabelIdx] : $"label_{span.LabelIdx}";
                string placeholder = $"<{label.ToUpper()}>";
                int start = span.Start + offset;
                int end = span.End + offset;

                string before = redactedText.Substring(0, start);
                string after = redactedText.Substring(end);

                redactedText = before + placeholder + after;
                offset += placeholder.Length - (span.End - span.Start);
            }

            Console.WriteLine($"\nRedacted text: {redactedText}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
