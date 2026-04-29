using System;
using System.Collections.Generic;
using System.IO;
using Opf.Core.Weights;

namespace Opf.Core.Model;

public static class TransformerModelFactory
{
    public static TransformerModel Create(string checkpointPath)
    {
        var configPath = Path.Combine(checkpointPath, "config.json");
        var config = ModelConfig.Load(configPath);
        var loader = new CheckpointLoader(checkpointPath);

        // Embeddings
        float[] embedWeight = loader.GetTensor("embedding.weight");
        var embeddings = new EmbeddingsLayer(embedWeight, config.VocabSize, config.HiddenSize);

        // Blocks
        var blocks = new List<TransformerBlock>();
        var rope = new RotaryEmbedding(
            config.HiddenSize / config.NumAttentionHeads,
            config.RopeParams.RopeTheta,
            config.RopeParams.OriginalMaxPositionEmbeddings
        );

        int headDim = config.HiddenSize / config.NumAttentionHeads;
        int numExperts = config.NumExperts;
        int topK = config.ExpertsPerToken;
        int hiddenSize = config.HiddenSize;
        int intermediateSize = config.IntermediateSize;

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            // Attention Norm
            float[] attnNormWeight = loader.GetTensor($"block.{i}.attn.norm.weight");
            var attnNorm = new RMSNormLayer(attnNormWeight, config.RmsNormEps);

            // Attention
            float[] wq = loader.GetTensor($"block.{i}.attn.q_proj.weight");
            float[] wk = loader.GetTensor($"block.{i}.attn.k_proj.weight");
            float[] wv = loader.GetTensor($"block.{i}.attn.v_proj.weight");
            float[] wo = loader.GetTensor($"block.{i}.attn.out_proj.weight");
            var attn = new GroupedQueryAttention(wq, wk, wv, wo, config.HiddenSize, config.NumAttentionHeads, config.NumKeyValueHeads, rope);

            // FFN Norm
            float[] ffnNormWeight = loader.GetTensor($"block.{i}.mlp.norm.weight");
            var ffnNorm = new RMSNormLayer(ffnNormWeight, config.RmsNormEps);

            // MoE
            float[] gateWeight = loader.GetTensor($"block.{i}.mlp.router.weight");

            float[][] expertW1 = new float[numExperts][];
            float[][] expertW2 = new float[numExperts][];
            float[][] expertW3 = new float[numExperts][];

            // Python maps:
            // block.{n}.mlp.swiglu.weight.blocks -> block.{n}.mlp.mlp1_weight
            // This tensor is size [numExperts, hiddenSize, intermediateSize * 2] originally? Or [numExperts, intermediateSize * 2, hiddenSize].
            // Mxfp4Unquantizer unquantizes it into linear floats.

            // From Python checkpoint mapping logic: out has shape (rows_total, B * 2) and gets reshaped to prefix_shape, G, B*2
            // Let's rely on Mxfp4 tensor unquantization with rows = numExperts * intermediateSize * 2, colsBytes = hiddenSize / 2

            float[] mlp1_fused = loader.GetMxfp4Tensor($"block.{i}.mlp.swiglu.weight", numExperts * intermediateSize * 2, hiddenSize / 2);
            float[] mlp2 = loader.GetMxfp4Tensor($"block.{i}.mlp.out.weight", numExperts * hiddenSize, intermediateSize / 2);

            // Need to split fused mlp1 to W1, W3. Python splits on the last dimension typically, or intermediate size.
            // Wait, looking at python: out[..., :intermediate_size] for W1... if it's concatenated along the intermediate dim, then it's [numExperts, 2*intermediateSize, hiddenSize]

            int mlp1_expert_size = 2 * intermediateSize * hiddenSize;
            int mlp2_expert_size = hiddenSize * intermediateSize;

            for(int e=0; e<numExperts; e++)
            {
                expertW1[e] = new float[hiddenSize * intermediateSize];
                expertW3[e] = new float[hiddenSize * intermediateSize];
                expertW2[e] = new float[hiddenSize * intermediateSize];

                int eOffset1 = e * mlp1_expert_size;
                // Splitting SwiGLU block. In python, it splits intermediate dimension.
                // Assuming layout [numExperts, 2 * intermediateSize, hiddenSize]
                // Wait, if it is [numExperts, hiddenSize, 2 * intermediateSize]?
                // Usually PyTorch linear is [out_features, in_features]
                // mlp1 out_features is 2*intermediateSize, in_features is hiddenSize
                // So [numExperts, 2*intermediateSize, hiddenSize]

                for(int r=0; r<intermediateSize; r++)
                {
                    // W1 is first half of out_features
                    Array.Copy(mlp1_fused, eOffset1 + r * hiddenSize, expertW1[e], r * hiddenSize, hiddenSize);

                    // W3 is second half of out_features
                    Array.Copy(mlp1_fused, eOffset1 + (intermediateSize + r) * hiddenSize, expertW3[e], r * hiddenSize, hiddenSize);
                }

                int eOffset2 = e * mlp2_expert_size;
                Array.Copy(mlp2, eOffset2, expertW2[e], 0, mlp2_expert_size);
            }

            var moe = new SparseMoE(gateWeight, expertW1, expertW2, expertW3, numExperts, topK, hiddenSize, intermediateSize);

            blocks.Add(new TransformerBlock(attnNorm, attn, ffnNorm, moe));
        }

        // Final Norm
        float[] finalNormWeight = loader.GetTensor("norm.weight");
        var finalNorm = new RMSNormLayer(finalNormWeight, config.RmsNormEps);

        // Classifier
        float[] classifierWeight = loader.GetTensor("unembedding.weight");

        return new TransformerModel(embeddings, blocks, finalNorm, classifierWeight, config.HiddenSize, config.NumLabels);
    }
}
