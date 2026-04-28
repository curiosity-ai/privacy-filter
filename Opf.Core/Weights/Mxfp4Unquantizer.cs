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
        // `columns` represents the number of bytes per row.
        // The output array will have 2 floats for each byte in the block.
        float[] outFloats = new float[rows * columns * 2];

        for (int r = 0; r < rows; r++)
        {
            int exp = scales[r] - 127;
            for (int c = 0; c < columns; c++)
            {
                int blockIdx = r * columns + c;
                byte blk = blocks[blockIdx];

                int idx_lo = blk & 0x0F;
                int idx_hi = (blk >> 4) & 0x0F;

                int outIdx = (r * columns * 2) + (c * 2);

                outFloats[outIdx] = MathF.ScaleB(Fp4Values[idx_lo], exp);
                outFloats[outIdx + 1] = MathF.ScaleB(Fp4Values[idx_hi], exp);
            }
        }

        return outFloats;
    }
}
