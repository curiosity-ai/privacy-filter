namespace Opf.Core;

public class ModelConfig
{
    public string ModelType { get; set; } = "privacy_filter";
    public int NumHiddenLayers { get; set; } = 36;
    public int NumExperts { get; set; } = 128;
    public int ExpertsPerToken { get; set; } = 4;
    public int VocabSize { get; set; } = 201088;
    public int? NumLabels { get; set; } = null;
    public int HiddenSize { get; set; } = 2880;
    public int IntermediateSize { get; set; } = 2880;
    public float SwigluLimit { get; set; } = 7.0f;
    public bool PackedGeglu { get; set; } = false;
    public int HeadDim { get; set; } = 64;
    public int NumAttentionHeads { get; set; } = 64;
    public int NumKeyValueHeads { get; set; } = 8;
    public int SlidingWindow { get; set; } = 128;
    public bool BidirectionalContext { get; set; } = false;
    public int BidirectionalLeftContext { get; set; } = 0;
    public int BidirectionalRightContext { get; set; } = 0;
    public int InitialContextLength { get; set; } = 4096;
    public float RopeTheta { get; set; } = 150000.0f;
    public float RopeScalingFactor { get; set; } = 32.0f;
    public float RopeNtkAlpha { get; set; } = 1.0f;
    public float RopeNtkBeta { get; set; } = 32.0f;
    public int TorchOpsBatch { get; set; } = 32;
    public string ParamDtype { get; set; } = "bfloat16";
}
