using System;
using System.IO;
using System.Text.Json;
using Xunit;
using Opf.Core.Model;
using Opf.Core.Decoding;
using System.Linq;

namespace Opf.Tests;

public class ModelParityTests
{
    private class RmsNormData
    {
        public float[] input { get; set; } = Array.Empty<float>();
        public float[] weight { get; set; } = Array.Empty<float>();
        public float eps { get; set; }
        public int seq_len { get; set; }
        public int hidden_size { get; set; }
        public float[] output { get; set; } = Array.Empty<float>();
    }

    private class GqaData
    {
        public float[] input { get; set; } = Array.Empty<float>();
        public float[] wq { get; set; } = Array.Empty<float>();
        public float[] wk { get; set; } = Array.Empty<float>();
        public float[] wv { get; set; } = Array.Empty<float>();
        public float[] wo { get; set; } = Array.Empty<float>();
        public int seq_len { get; set; }
        public int hidden_size { get; set; }
        public int num_heads { get; set; }
        public int num_kv_heads { get; set; }
        public float[] output { get; set; } = Array.Empty<float>();
    }

    private class MoeData
    {
        public float[] input { get; set; } = Array.Empty<float>();
        public float[] gate_weight { get; set; } = Array.Empty<float>();
        public float[][] expert_w1 { get; set; } = Array.Empty<float[]>();
        public float[][] expert_w2 { get; set; } = Array.Empty<float[]>();
        public float[][] expert_w3 { get; set; } = Array.Empty<float[]>();
        public int seq_len { get; set; }
        public int hidden_size { get; set; }
        public int intermediate_size { get; set; }
        public int num_experts { get; set; }
        public int top_k { get; set; }
        public float[] output { get; set; } = Array.Empty<float>();
    }

    private class ViterbiData
    {
        public float[] logits { get; set; } = Array.Empty<float>();
        public float[] start_transitions { get; set; } = Array.Empty<float>();
        public float[] transitions { get; set; } = Array.Empty<float>();
        public float[] end_transitions { get; set; } = Array.Empty<float>();
        public int seq_len { get; set; }
        public int num_labels { get; set; }
        public int[] output { get; set; } = Array.Empty<int>();
    }

    private void AssertArrayClose(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"Mismatch at index {i}: expected {expected[i]}, got {actual[i]} (diff: {Math.Abs(expected[i] - actual[i])})");
        }
    }

    [Fact]
    public void TestRMSNormParity()
    {
        string jsonString = File.ReadAllText("../../../rmsnorm_data.json");
        var data = JsonSerializer.Deserialize<RmsNormData>(jsonString)!;

        var rmsnorm = new RMSNormLayer(data.weight, data.eps);

        float[] csharpOutput = new float[data.input.Length];

        for (int i = 0; i < data.seq_len; i++)
        {
            var inputSlice = data.input.AsSpan(i * data.hidden_size, data.hidden_size);
            var outputSlice = csharpOutput.AsSpan(i * data.hidden_size, data.hidden_size);
            rmsnorm.Forward(inputSlice, outputSlice);
        }

        AssertArrayClose(data.output, csharpOutput);
    }

    [Fact]
    public void TestGQAParity()
    {
        string jsonString = File.ReadAllText("../../../gqa_data.json");
        var data = JsonSerializer.Deserialize<GqaData>(jsonString)!;

        int headDim = data.hidden_size / data.num_heads;
        var rope = new RotaryEmbedding(headDim);

        var gqa = new GroupedQueryAttention(
            data.wq, data.wk, data.wv, data.wo,
            data.hidden_size, data.num_heads, data.num_kv_heads, rope
        );

        float[] csharpOutput = new float[data.output.Length];
        gqa.Forward(data.input, csharpOutput, data.seq_len);

        AssertArrayClose(data.output, csharpOutput, 2e-3f);
    }

    [Fact]
    public void TestMoEParity()
    {
        string jsonString = File.ReadAllText("../../../moe_data.json");
        var data = JsonSerializer.Deserialize<MoeData>(jsonString)!;

        var moe = new SparseMoE(
            data.gate_weight, data.expert_w1, data.expert_w2, data.expert_w3,
            data.num_experts, data.top_k, data.hidden_size, data.intermediate_size
        );

        float[] csharpOutput = new float[data.output.Length];
        moe.Forward(data.input, csharpOutput, data.seq_len);

        AssertArrayClose(data.output, csharpOutput, 2e-3f);
    }

    [Fact]
    public void TestViterbiParity()
    {
        string jsonString = File.ReadAllText("../../../viterbi_data.json");
        var data = JsonSerializer.Deserialize<ViterbiData>(jsonString)!;

        var viterbi = new ViterbiCRFDecoder(
            data.num_labels, data.start_transitions, data.transitions, data.end_transitions
        );

        var csharpOutput = viterbi.Decode(data.logits, data.seq_len);

        Assert.Equal(data.output, csharpOutput);
    }
}
