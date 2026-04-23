using System;
using System.IO;

namespace Opf.Core.Weights;

public static class Mxfp4Unquantizer
{
    private static readonly float[] Fp4Values = {
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    public static float[] Unquantize(byte[] blocks, int[] scales, int rows, int columns)
    {
        // For OPF, blocks is MXFP4. scales are uint8 offsets by 127
        // Let's defer full implementation until we know exact shapes,
        // but this is the skeleton matching Python `_get_mxfp4_tensor` logic.
        throw new NotImplementedException("MXFP4 to Float32/BFloat16 mapping to be completed.");
    }
}
