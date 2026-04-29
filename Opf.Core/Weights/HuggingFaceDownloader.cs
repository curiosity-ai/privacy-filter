using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Opf.Core.Weights;

public class HuggingFaceDownloader
{
    private readonly HttpClient _httpClient;
    private const string DefaultRepo = "openai/privacy-filter";

    public HuggingFaceDownloader()
    {
        _httpClient = new HttpClient();
    }

    public async Task<string> EnsureModelDownloadedAsync(string? targetDirectory = null)
    {
        targetDirectory ??= Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".opf",
            "privacy_filter"
        );

        Directory.CreateDirectory(targetDirectory);

        string[] requiredFiles = {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "viterbi_calibration.json"
        };

        foreach (var file in requiredFiles)
        {
            var filePath = Path.Combine(targetDirectory, file);
            if (!File.Exists(filePath))
            {
                var url = $"https://huggingface.co/{DefaultRepo}/resolve/main/{file}";
                Console.WriteLine($"Downloading {file} from {url}...");

                try
                {
                    using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();

                    using var stream = await response.Content.ReadAsStreamAsync();
                    using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
                    await stream.CopyToAsync(fileStream);
                    Console.WriteLine($"Downloaded {file} successfully.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error downloading {file}: {ex.Message}");
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                    }
                    throw;
                }
            }
        }

        return targetDirectory;
    }
}
