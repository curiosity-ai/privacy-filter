using System;
using System.IO;
using System.Text.Json;

namespace OpfPort.Core
{
    public class ConfigLoader
    {
        public static ModelConfig LoadConfig(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found at {path}");

            var json = File.ReadAllText(path);
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var config = new ModelConfig();

            if (root.TryGetProperty("hidden_size", out var dModel))
                config.d_model = dModel.GetInt32();

            if (root.TryGetProperty("num_hidden_layers", out var nLayers))
                config.n_layers = nLayers.GetInt32();

            if (root.TryGetProperty("num_attention_heads", out var nHeads))
                config.n_heads = nHeads.GetInt32();

            if (root.TryGetProperty("num_key_value_heads", out var nKvHeads))
                config.n_kv_heads = nKvHeads.GetInt32();

            if (root.TryGetProperty("max_position_embeddings", out var maxSeqLen))
                config.max_seq_len = maxSeqLen.GetInt32();

            if (root.TryGetProperty("num_experts", out var nExperts))
                config.n_experts = nExperts.GetInt32();

            if (root.TryGetProperty("num_experts_per_tok", out var nExpertsPerTok))
                config.n_experts_per_tok = nExpertsPerTok.GetInt32();

            if (root.TryGetProperty("intermediate_size", out var dFf))
                config.d_ff = dFf.GetInt32();

            if (root.TryGetProperty("vocab_size", out var vocabSize))
                config.vocab_size = vocabSize.GetInt32();

            if (root.TryGetProperty("rms_norm_eps", out var normEps))
                config.norm_eps = normEps.GetSingle();

            return config;
        }
    }
}
