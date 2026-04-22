using System;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;

namespace OpfPort.Core
{
    public class SafetensorsLoader
    {
        public class TensorInfo
        {
            public string dtype { get; set; } = string.Empty;
            public long[] shape { get; set; } = Array.Empty<long>();
            public long[] data_offsets { get; set; } = Array.Empty<long>();
        }

        public static Dictionary<string, TensorInfo> LoadHeader(string path, out ulong headerLength)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fs);
            headerLength = reader.ReadUInt64();
            byte[] headerBytes = reader.ReadBytes((int)headerLength);
            var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);
            var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerJson);

            var result = new Dictionary<string, TensorInfo>();
            if (dict != null) {
                foreach (var kvp in dict)
                {
                    if (kvp.Key == "__metadata__") continue;
                    var info = JsonSerializer.Deserialize<TensorInfo>(kvp.Value.GetRawText());
                    if (info != null)
                    {
                        result[kvp.Key] = info;
                    }
                }
            }
            return result;
        }

        public static float[] LoadTensor(string path, TensorInfo info, ulong headerLength)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            long offset = 8 + (long)headerLength + info.data_offsets[0];
            fs.Seek(offset, SeekOrigin.Begin);

            long byteLength = info.data_offsets[1] - info.data_offsets[0];
            byte[] buffer = new byte[byteLength];
            int read = 0;
            while (read < byteLength)
            {
                int bytesRead = fs.Read(buffer, read, (int)(byteLength - read));
                if (bytesRead == 0) break;
                read += bytesRead;
            }

            if (info.dtype == "BF16")
            {
                // Convert BF16 to FP32
                float[] floats = new float[byteLength / 2];
                for (int i = 0; i < floats.Length; i++)
                {
                    ushort bf16 = BitConverter.ToUInt16(buffer, i * 2);
                    uint fp32 = (uint)bf16 << 16;
                    floats[i] = BitConverter.Int32BitsToSingle((int)fp32);
                }
                return floats;
            }
            else if (info.dtype == "F32")
            {
                float[] floats = new float[byteLength / 4];
                Buffer.BlockCopy(buffer, 0, floats, 0, (int)byteLength);
                return floats;
            }
            else
            {
                throw new NotSupportedException($"Dtype {info.dtype} not supported yet.");
            }
        }
    }
}
