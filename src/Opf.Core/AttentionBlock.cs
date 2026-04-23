using System;
using System.Numerics.Tensors;

namespace Opf.Core;

public class AttentionBlock
{
    private readonly int _hiddenSize;
    private readonly int _headDim;
    private readonly int _numAttentionHeads;
    private readonly int _numKeyValueHeads;
    private readonly RMSNorm _norm;
    private readonly float _qkScale;

    // Weights:
    public float[] QkvWeight { get; set; } = Array.Empty<float>();
    public float[] QkvBias { get; set; } = Array.Empty<float>();
    public float[] OutWeight { get; set; } = Array.Empty<float>();
    public float[] OutBias { get; set; } = Array.Empty<float>();
    public float[] Sinks { get; set; } = Array.Empty<float>();

    public RotaryEmbedding Rope { get; }

    public AttentionBlock(ModelConfig config)
    {
        _hiddenSize = config.HiddenSize;
        _headDim = config.HeadDim;
        _numAttentionHeads = config.NumAttentionHeads;
        _numKeyValueHeads = config.NumKeyValueHeads;
        _norm = new RMSNorm(config.HiddenSize);

        _qkScale = 1.0f / (float)Math.Sqrt(Math.Sqrt(config.HeadDim));

        Rope = new RotaryEmbedding(
            headDim: config.HeadDim,
            baseVal: config.RopeTheta,
            initialContextLength: config.InitialContextLength,
            scalingFactor: config.RopeScalingFactor,
            ntkAlpha: config.RopeNtkAlpha,
            ntkBeta: config.RopeNtkBeta
        );
    }
}
