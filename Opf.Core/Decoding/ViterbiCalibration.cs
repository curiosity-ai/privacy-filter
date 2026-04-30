using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Opf.Core.Decoding;

public class ViterbiCalibrationBiases
{
    [JsonPropertyName("transition_bias_background_stay")]
    public float TransitionBiasBackgroundStay { get; set; } = 0.0f;

    [JsonPropertyName("transition_bias_background_to_start")]
    public float TransitionBiasBackgroundToStart { get; set; } = 0.0f;

    [JsonPropertyName("transition_bias_inside_to_continue")]
    public float TransitionBiasInsideToContinue { get; set; } = 0.0f;

    [JsonPropertyName("transition_bias_inside_to_end")]
    public float TransitionBiasInsideToEnd { get; set; } = 0.0f;

    [JsonPropertyName("transition_bias_end_to_background")]
    public float TransitionBiasEndToBackground { get; set; } = 0.0f;

    [JsonPropertyName("transition_bias_end_to_start")]
    public float TransitionBiasEndToStart { get; set; } = 0.0f;
}

public class ViterbiDefaultOperatingPoint
{
    [JsonPropertyName("biases")]
    public ViterbiCalibrationBiases Biases { get; set; } = new();
}

public class ViterbiOperatingPoints
{
    [JsonPropertyName("default")]
    public ViterbiDefaultOperatingPoint Default { get; set; } = new();
}

public class ViterbiCalibration
{
    [JsonPropertyName("operating_points")]
    public ViterbiOperatingPoints OperatingPoints { get; set; } = new();

    public static ViterbiCalibration Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"viterbi_calibration.json not found: {path}");
        }

        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<ViterbiCalibration>(json) ?? new ViterbiCalibration();
    }
}
