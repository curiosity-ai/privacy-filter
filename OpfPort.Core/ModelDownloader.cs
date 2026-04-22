using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace OpfPort.Core
{
    public class ModelDownloader
    {
        private const string RepoId = "openai/privacy-filter";
        private const string BaseUrl = "https://huggingface.co";

        public static async Task DownloadModelAsync(string outputDir)
        {
            Directory.CreateDirectory(outputDir);
            using var client = new HttpClient();

            string[] files = new[]
            {
                "config.json",
                "model.safetensors",
                "tokenizer.json"
            };

            foreach (var file in files)
            {
                string url = $"{BaseUrl}/{RepoId}/resolve/main/{file}";
                string outputPath = Path.Combine(outputDir, file);

                if (!File.Exists(outputPath))
                {
                    Console.WriteLine($"Downloading {file}...");
                    var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    if (response.IsSuccessStatusCode) {
                        using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.None);
                        await response.Content.CopyToAsync(fs);
                        Console.WriteLine($"Downloaded {file}");
                    } else {
                        Console.WriteLine($"Could not download {file}: {response.StatusCode}");
                    }
                }
                else
                {
                    Console.WriteLine($"{file} already exists.");
                }
            }
        }
    }
}
