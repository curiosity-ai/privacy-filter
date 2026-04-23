using System;
using System.Collections.Generic;

namespace Opf.Core;

public class PredictionResult
{
    public string Text { get; set; } = string.Empty;
    public List<SpanItem> Spans { get; set; } = new();
}

public class InferenceRuntime
{
    private readonly Transformer _model;
    private readonly ViterbiCRFDecoder _decoder;
    private readonly Tokenizer _tokenizer;

    public InferenceRuntime(Transformer model, ViterbiCRFDecoder decoder, Tokenizer tokenizer)
    {
        _model = model;
        _decoder = decoder;
        _tokenizer = tokenizer;
    }

    public PredictionResult PredictText(string text)
    {
        // 1. Tokenize
        var inputIdsList = _tokenizer.Encode(text);
        int[] inputIds = inputIdsList.ToArray();

        // 2. Run model forward pass (Mocking tensor outputs for now, full forward pass requires extensive tensor math)
        int seqLen = inputIds.Length;
        int numClasses = _model.Config.NumLabels ?? 33;
        float[] logits = new float[seqLen * numClasses];

        // 3. Decode
        var decodedLabels = _decoder.Decode(logits, seqLen);

        // 4. Resolve spans
        var spans = Spans.LabelsToSpans(decodedLabels);

        return new PredictionResult { Text = text, Spans = spans };
    }
}
