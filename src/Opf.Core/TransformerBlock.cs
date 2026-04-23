using System;

namespace Opf.Core;

public class TransformerBlock
{
    public AttentionBlock Attention { get; }
    public MLPBlock Mlp { get; }

    public TransformerBlock(ModelConfig config, int layerIdx)
    {
        Attention = new AttentionBlock(config);
        Mlp = new MLPBlock(config);
    }
}
