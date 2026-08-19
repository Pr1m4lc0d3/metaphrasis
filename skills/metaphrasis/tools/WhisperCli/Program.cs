using System.Globalization;
using System.Text;
using System.Text.Json;
using Whisper.net;

namespace WhisperCli;

/// <summary>
/// Minimal Whisper host: 16 kHz mono WAV in, timestamped JSON segments out.
///
/// Exists because the Python route is unavailable on this machine — transformers
/// pulls in sklearn and pandas, and pandas there is built against an older numpy
/// ABI. Rather than repair a shared environment Deliberon depends on, this calls
/// the same Whisper.net version the app already ships.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        if (args.Length < 2)
        {
            Console.Error.WriteLine(
                "usage: whisper-cli <model.bin> <audio-16k-mono.wav> [vocabulary prompt]");
            return 2;
        }

        var modelPath = args[0];
        var audioPath = args[1];

        // Whisper reliably mangles unusual proper nouns — "Deliberon" comes back
        // as "Deliveron", "VoxCPM2" as "Box CPM2". Seeding the decoder with the
        // real spellings biases it toward them, which matters when the transcript
        // is going to be checked against approved copy.
        var prompt = args.Length > 2 ? args[2] : null;

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"model not found: {modelPath}");
            return 3;
        }

        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"audio not found: {audioPath}");
            return 4;
        }

        try
        {
            using var factory = WhisperFactory.FromPath(modelPath);
            var builder = factory.CreateBuilder().WithLanguage("en");
            if (!string.IsNullOrWhiteSpace(prompt))
            {
                builder = builder.WithPrompt(prompt);
            }

            using var processor = builder.Build();

            await using var audio = File.OpenRead(audioPath);

            var segments = new List<object>();
            await foreach (var segment in processor.ProcessAsync(audio))
            {
                segments.Add(new
                {
                    start = Math.Round(segment.Start.TotalSeconds, 2),
                    end = Math.Round(segment.End.TotalSeconds, 2),
                    text = segment.Text.Trim(),
                });
            }

            var payload = new
            {
                model = Path.GetFileName(modelPath),
                audio = Path.GetFileName(audioPath),
                segment_count = segments.Count,
                text = string.Join(" ", segments.Select(s =>
                    ((dynamic)s).text as string).Where(t => !string.IsNullOrWhiteSpace(t))),
                segments,
            };

            Console.WriteLine(JsonSerializer.Serialize(payload,
                new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"transcription failed: {ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }
}
