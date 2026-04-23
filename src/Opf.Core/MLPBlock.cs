using System;

namespace Opf.Core;

public class MLPBlock
{
    private readonly int _numExperts;
    private readonly int _expertsPerToken;
    private readonly RMSNorm _norm;

    public float[] GateWeight { get; set; } = Array.Empty<float>();
    public float[] GateBias { get; set; } = Array.Empty<float>();

    public float[] Mlp1Weight { get; set; } = Array.Empty<float>();
    public float[] Mlp1Bias { get; set; } = Array.Empty<float>();

    public float[] Mlp2Weight { get; set; } = Array.Empty<float>();
    public float[] Mlp2Bias { get; set; } = Array.Empty<float>();

    public MLPBlock(ModelConfig config)
    {
        _numExperts = config.NumExperts;
        _expertsPerToken = config.ExpertsPerToken;
        _norm = new RMSNorm(config.HiddenSize);
    }
}
