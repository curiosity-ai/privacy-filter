using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace OpfPort.Core
{
    public class ModelConfig
    {
        public int d_model { get; set; } = 640;
        public int n_layers { get; set; } = 8;
        public int n_heads { get; set; } = 14;
        public int n_kv_heads { get; set; } = 2;
        public int max_seq_len { get; set; } = 128000;
        public int n_experts { get; set; } = 128;
        public int n_experts_per_tok { get; set; } = 4;
        public int d_ff { get; set; } = 512;
        public int vocab_size { get; set; } = 100256;
        public float norm_eps { get; set; } = 1e-5f;
    }

    public class RMSNorm
    {
        private float[] _weight;
        private float _eps;

        public RMSNorm(int dim, float eps, float[] weight)
        {
            _weight = weight ?? new float[dim];
            if (weight == null) Array.Fill(_weight, 1.0f);
            _eps = eps;
        }

        public void Forward(ReadOnlySpan<float> input, Span<float> output)
        {
            float variance = TensorPrimitives.SumOfSquares(input) / input.Length;
            float invRoot = 1.0f / (float)Math.Sqrt(variance + _eps);
            TensorPrimitives.Multiply(input, invRoot, output);
            TensorPrimitives.Multiply(output, _weight, output);
        }
    }

    public class Linear
    {
        private float[] _weight;
        public int InFeatures { get; }
        public int OutFeatures { get; }

        public Linear(int inFeatures, int outFeatures, float[] weight)
        {
            InFeatures = inFeatures;
            OutFeatures = outFeatures;
            _weight = weight ?? new float[inFeatures * outFeatures];
        }

        public void Forward(ReadOnlySpan<float> input, Span<float> output)
        {
            for (int o = 0; o < OutFeatures; o++)
            {
                output[o] = TensorPrimitives.Dot(input, new ReadOnlySpan<float>(_weight, o * InFeatures, InFeatures));
            }
        }
    }

    public class Embedding
    {
        private float[] _weight;
        public int NumEmbeddings { get; }
        public int EmbeddingDim { get; }

        public Embedding(int numEmbeddings, int embeddingDim, float[] weight)
        {
            NumEmbeddings = numEmbeddings;
            EmbeddingDim = embeddingDim;
            _weight = weight ?? new float[numEmbeddings * embeddingDim];
        }

        public void Forward(int inputId, Span<float> output)
        {
            var emb = new ReadOnlySpan<float>(_weight, inputId * EmbeddingDim, EmbeddingDim);
            emb.CopyTo(output);
        }
    }

    public class AttentionBlock
    {
        public Linear q_proj;
        public Linear k_proj;
        public Linear v_proj;
        public Linear o_proj;
        private int _d_model;
        private int _n_heads;
        private int _n_kv_heads;

        public AttentionBlock(int d_model, int n_heads, int n_kv_heads)
        {
            _d_model = d_model;
            _n_heads = n_heads;
            _n_kv_heads = n_kv_heads;
            int head_dim = d_model / n_heads;
            q_proj = new Linear(d_model, n_heads * head_dim, null);
            k_proj = new Linear(d_model, n_kv_heads * head_dim, null);
            v_proj = new Linear(d_model, n_kv_heads * head_dim, null);
            o_proj = new Linear(n_heads * head_dim, d_model, null);
        }

        public void Forward(Span<float> state) {
            // Simplified pass-through for compilation
        }
    }

    public class MLPBlock
    {
        public Linear gate_proj;
        public Linear up_proj;
        public Linear down_proj;

        public MLPBlock(int d_model, int d_ff)
        {
            gate_proj = new Linear(d_model, d_ff, null);
            up_proj = new Linear(d_model, d_ff, null);
            down_proj = new Linear(d_ff, d_model, null);
        }

        public void Forward(Span<float> state) {
            // Simplified pass-through
        }
    }

    public class TransformerBlock
    {
        public RMSNorm pre_norm;
        public RMSNorm post_norm;
        public AttentionBlock attn;
        public MLPBlock mlp;

        public TransformerBlock(ModelConfig config)
        {
            pre_norm = new RMSNorm(config.d_model, config.norm_eps, null);
            post_norm = new RMSNorm(config.d_model, config.norm_eps, null);
            attn = new AttentionBlock(config.d_model, config.n_heads, config.n_kv_heads);
            mlp = new MLPBlock(config.d_model, config.d_ff);
        }

        public void Forward(Span<float> state) {
            Span<float> temp = new float[state.Length];
            pre_norm.Forward(state, temp);
            attn.Forward(temp);
            // residual
            for(int i = 0; i < state.Length; i++) state[i] += temp[i];

            post_norm.Forward(state, temp);
            mlp.Forward(temp);
            for(int i = 0; i < state.Length; i++) state[i] += temp[i];
        }
    }

    public class Transformer
    {
        public ModelConfig Config { get; }
        public Embedding embedding;
        public RMSNorm final_norm;
        public Linear head;
        public TransformerBlock[] blocks;

        public Transformer(ModelConfig config, Dictionary<string, float[]> weights = null)
        {
            Config = config;
            embedding = new Embedding(config.vocab_size, config.d_model, null);
            final_norm = new RMSNorm(config.d_model, config.norm_eps, null);
            head = new Linear(config.d_model, 33, null);
            blocks = new TransformerBlock[config.n_layers];
            for (int i = 0; i < config.n_layers; i++) {
                blocks[i] = new TransformerBlock(config);
            }
        }

        public float[] Forward(int[] tokens) {
            float[] logits = new float[tokens.Length * 33];
            float[] state = new float[Config.d_model];

            for (int i = 0; i < tokens.Length; i++) {
                embedding.Forward(tokens[i], state);
                foreach (var block in blocks) {
                    block.Forward(state);
                }
                final_norm.Forward(state, state);

                var logitSpan = new Span<float>(logits, i * 33, 33);
                head.Forward(state, logitSpan);
            }
            return logits;
        }
    }
}
