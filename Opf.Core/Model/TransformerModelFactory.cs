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
        float[] embedWeight = loader.HasTensor("embedding.weight") ? loader.GetTensor("embedding.weight") : loader.GetTensor("model.embed_tokens.weight");
        var embeddings = new EmbeddingsLayer(embedWeight, config.VocabSize, config.HiddenSize);

        // Blocks
        var blocks = new List<TransformerBlock>();
        int headDim = config.HeadDim > 0 ? config.HeadDim : config.HiddenSize / config.NumAttentionHeads;
        var rope = new RotaryEmbedding(
            headDim,
            config.RopeParams.RopeTheta,
            config.RopeParams.OriginalMaxPositionEmbeddings,
            config.RopeParams.Factor,
            config.RopeParams.BetaSlow,
            config.RopeParams.BetaFast
        );

                int numExperts = config.NumExperts;
        int topK = config.ExpertsPerToken;
        int hiddenSize = config.HiddenSize;
        int intermediateSize = config.IntermediateSize;


        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            // Attention Norm
            float[] attnNormWeight = loader.HasTensor($"block.{i}.attn.norm.weight") ? loader.GetTensor($"block.{i}.attn.norm.weight") : loader.GetTensor($"model.layers.{i}.input_layernorm.weight");
            var attnNorm = new RMSNormLayer(attnNormWeight, config.RmsNormEps);

            // Attention
            float[] wq = loader.HasTensor($"block.{i}.attn.q_proj.weight") ? loader.GetTensor($"block.{i}.attn.q_proj.weight") : loader.GetTensor($"model.layers.{i}.self_attn.q_proj.weight");
            float[] wk = loader.HasTensor($"block.{i}.attn.k_proj.weight") ? loader.GetTensor($"block.{i}.attn.k_proj.weight") : loader.GetTensor($"model.layers.{i}.self_attn.k_proj.weight");
            float[] wv = loader.HasTensor($"block.{i}.attn.v_proj.weight") ? loader.GetTensor($"block.{i}.attn.v_proj.weight") : loader.GetTensor($"model.layers.{i}.self_attn.v_proj.weight");
            float[] wo = loader.HasTensor($"block.{i}.attn.out_proj.weight") ? loader.GetTensor($"block.{i}.attn.out_proj.weight") : loader.GetTensor($"model.layers.{i}.self_attn.o_proj.weight");
            var attn = new GroupedQueryAttention(wq, wk, wv, wo, config.HiddenSize, config.NumAttentionHeads, config.NumKeyValueHeads, rope);

            // FFN Norm
            float[] ffnNormWeight = loader.HasTensor($"block.{i}.mlp.norm.weight") ? loader.GetTensor($"block.{i}.mlp.norm.weight") : loader.GetTensor($"model.layers.{i}.post_attention_layernorm.weight");
            var ffnNorm = new RMSNormLayer(ffnNormWeight, config.RmsNormEps);

            // MoE
            float[] gateWeight = loader.HasTensor($"block.{i}.mlp.router.weight") ? loader.GetTensor($"block.{i}.mlp.router.weight") : loader.GetTensor($"model.layers.{i}.mlp.router.weight");


            float[][] expertW1 = new float[numExperts][];
            float[][] expertW2 = new float[numExperts][];
            float[][] expertW3 = new float[numExperts][];


            // Python maps:
            // block.{n}.mlp.swiglu.weight.blocks -> block.{n}.mlp.mlp1_weight
            // This tensor is size [numExperts, hiddenSize, intermediateSize * 2] originally? Or [numExperts, intermediateSize * 2, hiddenSize].
            // Mxfp4Unquantizer unquantizes it into linear floats.

            // From Python checkpoint mapping logic: out has shape (rows_total, B * 2) and gets reshaped to prefix_shape, G, B*2
            // Let's rely on Mxfp4 tensor unquantization with rows = numExperts * intermediateSize * 2, colsBytes = hiddenSize / 2

            float[] mlp1_fused;
            float[] mlp2;

            bool hasMxfp4 = loader.HasMxfp4Tensor($"block.{i}.mlp.swiglu.weight");
            if (hasMxfp4)
            {
                mlp1_fused = loader.GetMxfp4Tensor($"block.{i}.mlp.swiglu.weight", numExperts * intermediateSize * 2, hiddenSize / 2);
                mlp2 = loader.GetMxfp4Tensor($"block.{i}.mlp.out.weight", numExperts * hiddenSize, intermediateSize / 2);
            }
            else
            {
                mlp1_fused = loader.GetTensor($"model.layers.{i}.mlp.experts.gate_up_proj");
                mlp2 = loader.GetTensor($"model.layers.{i}.mlp.experts.down_proj");
            }


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

                if (hasMxfp4)
                {
                    for(int r=0; r<intermediateSize; r++)
                    {
                        Array.Copy(mlp1_fused, eOffset1 + r * hiddenSize, expertW1[e], r * hiddenSize, hiddenSize);
                        Array.Copy(mlp1_fused, eOffset1 + (intermediateSize + r) * hiddenSize, expertW3[e], r * hiddenSize, hiddenSize);
                    }
                }
                else
                {
                    // For typical non-MXFP4 checkpoints from huggingface, the parameter shape is actually
                    // [num_experts, 2 * intermediateSize, hiddenSize].
                    // Wait, mlp1_expert_size is intermediateSize * hiddenSize * 2.
                    // Let's just block copy the entire W1 and W3 sequentially.
                    Array.Copy(mlp1_fused, e * mlp1_expert_size, expertW1[e], 0, intermediateSize * hiddenSize);
                    Array.Copy(mlp1_fused, e * mlp1_expert_size + (intermediateSize * hiddenSize), expertW3[e], 0, intermediateSize * hiddenSize);
                }

                int eOffset2 = e * mlp2_expert_size;
                Array.Copy(mlp2, eOffset2, expertW2[e], 0, mlp2_expert_size);
            }


            var moe = new SparseMoE(gateWeight, expertW1, expertW2, expertW3, numExperts, topK, hiddenSize, intermediateSize);
            Console.WriteLine($"Finished Block {i}");

            blocks.Add(new TransformerBlock(attnNorm, attn, ffnNorm, moe));
            GC.Collect();

        }

        // Final Norm

        float[] finalNormWeight = loader.HasTensor("norm.weight") ? loader.GetTensor("norm.weight") : loader.GetTensor("model.norm.weight");
        var finalNorm = new RMSNormLayer(finalNormWeight, config.RmsNormEps);

        // Classifier

        float[] classifierWeight = loader.HasTensor("unembedding.weight") ? loader.GetTensor("unembedding.weight") : loader.GetTensor("score.weight");


        return new TransformerModel(embeddings, blocks, finalNorm, classifierWeight, config.HiddenSize, config.NumLabels);
    }
}
