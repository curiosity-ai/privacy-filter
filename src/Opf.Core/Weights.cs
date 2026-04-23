using System;
using System.IO;

namespace Opf.Core;

public class Checkpoint : IDisposable
{
    private readonly SafeTensorsReader _reader;

    public Checkpoint(string path)
    {
        _reader = new SafeTensorsReader(path);
    }

    public bool Has(string name)
    {
        return _reader.HasTensor(name);
    }

    public float[] Get(string name)
    {
        return _reader.GetTensor(name);
    }

    public void Dispose()
    {
        _reader.Dispose();
    }
}
