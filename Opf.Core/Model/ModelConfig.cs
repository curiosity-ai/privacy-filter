using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Opf.Core.Model;

public class ModelConfig
{
    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 640;

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 640;

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; } = 8;

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; } = 14;

    [JsonPropertyName("num_key_value_heads")]
    public int NumKeyValueHeads { get; set; } = 2;

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 64;

    [JsonPropertyName("num_local_experts")]
    public int NumExperts { get; set; } = 128;

    [JsonPropertyName("num_experts_per_tok")]
    public int ExpertsPerToken { get; set; } = 4;

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 200064;

    [JsonPropertyName("id2label")]
    public Dictionary<string, string> Id2Label { get; set; } = new();

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; } = 1e-5f;

    [JsonPropertyName("rope_parameters")]
    public RopeParameters RopeParams { get; set; } = new();

    public int NumLabels => Id2Label.Count;

    public static ModelConfig Load(string path)
    {
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<ModelConfig>(json) ?? new ModelConfig();
    }
}

public class RopeParameters
{
    [JsonPropertyName("rope_theta")]
    public float RopeTheta { get; set; } = 150000.0f;

    [JsonPropertyName("original_max_position_embeddings")]
    public int OriginalMaxPositionEmbeddings { get; set; } = 4096;

    [JsonPropertyName("factor")]
    public float Factor { get; set; } = 32.0f;

    [JsonPropertyName("beta_slow")]
    public float BetaSlow { get; set; } = 1.0f;

    [JsonPropertyName("beta_fast")]
    public float BetaFast { get; set; } = 32.0f;
}
