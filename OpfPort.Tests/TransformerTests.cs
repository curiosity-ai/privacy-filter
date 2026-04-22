using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;
using Xunit;
using OpfPort.Core;

namespace OpfPort.Tests
{
    public class TransformerTests
    {
        [Fact]
        public void TestRMSNorm()
        {
            // Simple parity test placeholder
            var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var output = new float[4];

            var norm = new RMSNorm(4, 1e-5f, new float[] { 1.0f, 1.0f, 1.0f, 1.0f });
            norm.Forward(input, output);

            Assert.Equal(4, output.Length);
            // Assert close
        }

        [Fact]
        public void TestLinear()
        {
            var input = new float[] { 1.0f, 2.0f };
            var weight = new float[] { 0.5f, 0.5f, 1.0f, -1.0f }; // 2x2
            var linear = new Linear(2, 2, weight);

            var output = new float[2];
            linear.Forward(input, output);

            // Expected:
            // o[0] = 1 * 0.5 + 2 * 0.5 = 1.5
            // o[1] = 1 * 1.0 + 2 * -1.0 = -1.0
            Assert.Equal(1.5f, output[0]);
            Assert.Equal(-1.0f, output[1]);
        }
    }
}
