using System;
using System.Collections.Generic;

namespace Opf.Core;

public class Transformer
{
    public ModelConfig Config { get; }
    // Defer array allocations to prevent OOM on large model instantiations for testing
    public float[] EmbeddingWeight { get; set; } = Array.Empty<float>();
    public float[] TokenClassifierWeight { get; set; } = Array.Empty<float>();
    public float[] TokenClassifierBias { get; set; } = Array.Empty<float>();
    public RMSNorm Norm { get; }
    public List<TransformerBlock> Blocks { get; }

    public Transformer(ModelConfig config)
    {
        Config = config;
        Norm = new RMSNorm(config.HiddenSize);
        Blocks = new List<TransformerBlock>(config.NumHiddenLayers);

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            Blocks.Add(new TransformerBlock(config, i));
        }
    }
}
