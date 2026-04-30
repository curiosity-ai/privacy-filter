using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Opf.Core.Model;

namespace Opf.Core.Weights;

public class CheckpointLoader
{
    private readonly SafetensorsLoader _loader;

    public CheckpointLoader(string directoryPath)
    {
        _loader = new SafetensorsLoader(directoryPath);
    }

    public float[] GetTensor(string name)
    {
        if (name == "embedding.weight" && !HasTensor("embedding.weight")) name = "model.embed_tokens.weight";
        else if (name.Contains(".attn.norm.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".attn.norm.weight", ".input_layernorm.weight");
        else if (name.Contains(".attn.q_proj.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".attn.q_proj.weight", ".self_attn.q_proj.weight");
        else if (name.Contains(".attn.k_proj.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".attn.k_proj.weight", ".self_attn.k_proj.weight");
        else if (name.Contains(".attn.v_proj.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".attn.v_proj.weight", ".self_attn.v_proj.weight");
        else if (name.Contains(".attn.out_proj.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".attn.out_proj.weight", ".self_attn.o_proj.weight");
        else if (name.Contains(".mlp.norm.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".mlp.norm.weight", ".post_attention_layernorm.weight");
        else if (name.Contains(".mlp.router.weight") && !HasTensor(name)) name = name.Replace("block.", "model.layers.").Replace(".mlp.router.weight", ".mlp.router.weight");
        else if (name == "norm.weight" && !HasTensor("norm.weight")) name = "model.norm.weight";
        else if (name == "unembedding.weight" && !HasTensor("unembedding.weight")) name = "score.weight";

        byte[] bytes = _loader.GetTensorBytes(name);
        var meta = _loader.GetTensorMetadata(name).Meta;

        if (meta.Dtype == "BF16")
        {
            return ConvertBf16ToFp32(bytes);
        }
        else if (meta.Dtype == "F32")
        {
            float[] floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }

        throw new NotSupportedException($"Dtype {meta.Dtype} not supported for direct tensor loading.");
    }

    public float[] GetMxfp4Tensor(string name, int rows, int colsBytes)
    {
        if (name.Contains("mlp.swiglu.weight") && !HasTensor($"{name}.blocks")) name = name.Replace("block.", "model.layers.").Replace(".mlp.swiglu.weight", ".mlp.experts.gate_up_proj");
        if (name.Contains("mlp.out.weight") && !HasTensor($"{name}.blocks")) name = name.Replace("block.", "model.layers.").Replace(".mlp.out.weight", ".mlp.experts.down_proj");

        string blocksName = $"{name}.blocks";
        string scalesName = $"{name}.scales";

        if (!HasTensor(blocksName))
        {
            return GetTensor(name);
        }

        byte[] blocks = _loader.GetTensorBytes(blocksName);
        byte[] scalesBytes = _loader.GetTensorBytes(scalesName);

        // scales in python are uint8.
        int[] scales = new int[scalesBytes.Length];
        for (int i = 0; i < scalesBytes.Length; i++)
        {
            scales[i] = scalesBytes[i];
        }

        return Mxfp4Unquantizer.Unquantize(blocks, scales, rows, colsBytes);
    }

    public bool HasMxfp4Tensor(string name)
    {
        return _loader.HasTensor($"{name}.blocks") && _loader.HasTensor($"{name}.scales");
    }

    public bool HasTensor(string name)
    {
        return _loader.HasTensor(name);
    }

    private static float[] ConvertBf16ToFp32(byte[] bytes)
    {
        int numElements = bytes.Length / 2;
        float[] floats = new float[numElements];

        for (int i = 0; i < numElements; i++)
        {
            ushort bf16 = BitConverter.ToUInt16(bytes, i * 2);
            uint fp32 = (uint)bf16 << 16;
            floats[i] = BitConverter.Int32BitsToSingle((int)fp32);
        }
        return floats;
    }
}
