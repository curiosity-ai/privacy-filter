using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Runtime.InteropServices;
using Opf.Core.Weights;
using Opf.Core.Decoding;

namespace Opf.Core.Model;

public class ModelConfig
{
    public string model_type { get; set; } = "privacy_filter";
    public int num_hidden_layers { get; set; } = 36;
    public int num_experts { get; set; } = 128;
    public int experts_per_token { get; set; } = 4;
    public int vocab_size { get; set; } = 201088;
    public int? num_labels { get; set; }
    public int hidden_size { get; set; } = 2880;
    public int intermediate_size { get; set; } = 2880;
    public int head_dim { get; set; } = 64;
    public int num_attention_heads { get; set; } = 64;
    public int num_key_value_heads { get; set; } = 8;
    public int sliding_window { get; set; } = 128;
    public bool bidirectional_context { get; set; } = false;
    public int bidirectional_left_context { get; set; } = 0;
    public int bidirectional_right_context { get; set; } = 0;
    public int initial_context_length { get; set; } = 4096;
    public float rope_theta { get; set; } = 150000.0f;
    public float rope_scaling_factor { get; set; } = 32.0f;
    public float rope_ntk_alpha { get; set; } = 1.0f;
    public float rope_ntk_beta { get; set; } = 32.0f;
}

public static class ModelFactory
{
    public static (TransformerModel Model, ViterbiCRFDecoder Decoder) LoadFromDirectory(string directoryPath)
    {
        string configPath = Path.Combine(directoryPath, "config.json");
        string json = File.ReadAllText(configPath);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var config = JsonSerializer.Deserialize<ModelConfig>(json, options)
            ?? throw new InvalidOperationException("Failed to load config.json");

        var loader = new SafetensorsLoader(directoryPath);

        // Helper to get raw float array from bytes (assuming bfloat16 to float32 conversion logic might be needed,
        // but for now the OPF python mentions param_dtype = bfloat16. Let's do BFloat16 -> float cast if needed,
        // or just read as float if it's already F32.
        // NOTE: Actually OPF safetensors are in bfloat16, so we must convert bfloat16 to float32).
        float[] GetTensorAsFloat(string name)
        {
            var meta = loader.GetTensorMetadata(name);
            byte[] bytes = loader.GetTensorBytes(name);

            if (meta.Meta.Dtype == "BF16")
            {
                // Convert BF16 to FP32
                // BF16 is the upper 16 bits of a 32-bit float.
                int count = bytes.Length / 2;
                float[] floats = new float[count];
                var u32Span = MemoryMarshal.Cast<float, uint>(floats.AsSpan());
                for (int i = 0; i < count; i++)
                {
                    ushort bf16 = BitConverter.ToUInt16(bytes, i * 2);
                    u32Span[i] = (uint)bf16 << 16;
                }
                return floats;
            }
            else if (meta.Meta.Dtype == "F32")
            {
                int count = bytes.Length / 4;
                float[] floats = new float[count];
                Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
                return floats;
            }
            else
            {
                throw new NotSupportedException($"Dtype {meta.Meta.Dtype} not supported yet.");
            }
        }

        // Helper to unquantize MXFP4 block/scales
        float[] GetMxfp4Tensor(string blocksName, string scalesName, int rows, int cols)
        {
            byte[] blocksBytes = loader.GetTensorBytes(blocksName);
            byte[] scalesBytes = loader.GetTensorBytes(scalesName);
            // Scales are stored as uint8 but we treat as ints
            int[] scalesInt = new int[scalesBytes.Length];
            for (int i = 0; i < scalesBytes.Length; i++) scalesInt[i] = scalesBytes[i];

            return Mxfp4Unquantizer.Unquantize(blocksBytes, scalesInt, rows, cols);
        }

        // Load Embeddings
        float[] embeddingsWeight = GetTensorAsFloat("embedding.weight");
        var embeddings = new EmbeddingsLayer(embeddingsWeight, config.vocab_size, config.hidden_size);

        var blocks = new List<TransformerBlock>();

        var rope = new RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            config.initial_context_length,
            config.rope_scaling_factor,
            config.rope_ntk_alpha,
            config.rope_ntk_beta);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            float[] attnNormWeight = GetTensorAsFloat($"block.{i}.attn.norm.scale");
            var attnNorm = new RMSNormLayer(attnNormWeight);

            // Attention
            float[] wqkv = GetTensorAsFloat($"block.{i}.attn.qkv.weight");
            int qDim = config.num_attention_heads * config.head_dim;
            int kvDim = config.num_key_value_heads * config.head_dim;

            // OPF splits Q, K, V from a single QKV weight matrix
            // Q is size [hidden_size, qDim], K is [hidden_size, kvDim], V is [hidden_size, kvDim]
            // We need to slice it or just map it accordingly. Wait, linear layers in PyTorch are [out_features, in_features]
            // So wqkv is [qDim + 2*kvDim, hiddenSize]
            // We need wq [qDim, hiddenSize], wk [kvDim, hiddenSize], wv [kvDim, hiddenSize]
            // To work with our Matmul which expects right hand size to be [in_features, out_features]
            // we should transpose it or modify our Matmul. Let's assume we extract them.
            float[] Transpose(float[] src, int rows, int cols)
            {
                float[] dst = new float[rows * cols];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        dst[c * rows + r] = src[r * cols + c];
                return dst;
            }

            float[] wqTranspose = new float[qDim * config.hidden_size];
            float[] wkTranspose = new float[kvDim * config.hidden_size];
            float[] wvTranspose = new float[kvDim * config.hidden_size];

            for (int h = 0; h < config.hidden_size; h++)
            {
                for (int d = 0; d < qDim; d++)
                    wqTranspose[h * qDim + d] = wqkv[d * config.hidden_size + h];
                for (int d = 0; d < kvDim; d++)
                    wkTranspose[h * kvDim + d] = wqkv[(qDim + d) * config.hidden_size + h];
                for (int d = 0; d < kvDim; d++)
                    wvTranspose[h * kvDim + d] = wqkv[(qDim + kvDim + d) * config.hidden_size + h];
            }

            float[] woPyTorch = GetTensorAsFloat($"block.{i}.attn.out.weight");
            float[] wo = Transpose(woPyTorch, config.hidden_size, qDim);

            var gqa = new GroupedQueryAttention(wqTranspose, wkTranspose, wvTranspose, wo, config.hidden_size, config.num_attention_heads, config.num_key_value_heads, rope);

            // MLP / MoE
            float[] mlpNormWeight = GetTensorAsFloat($"block.{i}.mlp.norm.scale");
            var ffnNorm = new RMSNormLayer(mlpNormWeight);

            float[] gateWeightPyTorch = GetTensorAsFloat($"block.{i}.mlp.gate.weight");
            float[] gateWeight = Transpose(gateWeightPyTorch, config.num_experts, config.hidden_size);

            // Load MXFP4 weights
            // mlp1_weight corresponds to mlp.swiglu.weight.blocks and scales
            // Wait, Python _mxfp4_tensor reshape: out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
            // mlp1_weight shape in python: [num_experts, intermediate_size * 2, hidden_size] ??
            // Actually mlp1 is (num_experts, hidden_size, intermediate_size * 2) from model.py

            // Let's defer full slicing logic for now, just load it
            float[] w1w3Mxfp4 = GetMxfp4Tensor(
                $"block.{i}.mlp.swiglu.weight.blocks",
                $"block.{i}.mlp.swiglu.weight.scales",
                config.num_experts * (config.intermediate_size * 2), // rows
                config.hidden_size / 2 // cols
            );

            float[] w2Mxfp4 = GetMxfp4Tensor(
                $"block.{i}.mlp.out.weight.blocks",
                $"block.{i}.mlp.out.weight.scales",
                config.num_experts * config.hidden_size,
                config.intermediate_size / 2
            );

            // Reconstruct W1, W3, W2
            // To fully port this, we need the exact shapes, but to unblock parsing and construction:
            float[][] expertW1 = new float[config.num_experts][];
            float[][] expertW2 = new float[config.num_experts][];
            float[][] expertW3 = new float[config.num_experts][];

            for (int e = 0; e < config.num_experts; e++)
            {
                expertW1[e] = new float[config.hidden_size * config.intermediate_size];
                expertW2[e] = new float[config.intermediate_size * config.hidden_size];
                expertW3[e] = new float[config.hidden_size * config.intermediate_size];
            }

            var moe = new SparseMoE(gateWeight, expertW1, expertW2, expertW3, config.num_experts, config.experts_per_token, config.hidden_size, config.intermediate_size);

            blocks.Add(new TransformerBlock(attnNorm, gqa, ffnNorm, moe));
        }

        float[] finalNormWeight = GetTensorAsFloat("norm.scale");
        var finalNorm = new RMSNormLayer(finalNormWeight);

        int numClasses = config.num_labels ?? config.vocab_size;
        float[] unembeddingPyTorch = GetTensorAsFloat("unembedding.weight");
        float[] classifierWeight = new float[config.hidden_size * numClasses];
        for (int r = 0; r < numClasses; r++)
            for (int c = 0; c < config.hidden_size; c++)
                classifierWeight[c * numClasses + r] = unembeddingPyTorch[r * config.hidden_size + c];

        var model = new TransformerModel(embeddings, blocks, finalNorm, classifierWeight, config.hidden_size, numClasses);

        // Load CRF Decoder
        float[] startTransitions = GetTensorAsFloat("crf.start_transitions");
        float[] transitions = GetTensorAsFloat("crf.transitions");
        float[] endTransitions = GetTensorAsFloat("crf.end_transitions");

        var decoder = new ViterbiCRFDecoder(numClasses, startTransitions, transitions, endTransitions);

        return (model, decoder);
    }
}
