using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Opf.Core;

public static class CheckpointDownload
{
    private const string DefaultRepoId = "openai/privacy-filter";

    public static async Task<string> EnsureDefaultCheckpointAsync()
    {
        string homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string checkpointDir = Path.Combine(homeDir, ".opf", "privacy_filter");

        if (Directory.Exists(checkpointDir) && File.Exists(Path.Combine(checkpointDir, "model.safetensors")))
        {
            return checkpointDir;
        }

        Directory.CreateDirectory(checkpointDir);

        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(30);

        string[] filesToDownload = { "config.json", "model.safetensors", "viterbi_calibration.json" };

        foreach (var file in filesToDownload)
        {
            string url = $"https://huggingface.co/{DefaultRepoId}/resolve/main/{file}";
            Console.WriteLine($"Downloading {file}...");

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            using var contentStream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(Path.Combine(checkpointDir, file), FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);
            await contentStream.CopyToAsync(fileStream);
        }

        return checkpointDir;
    }
}
