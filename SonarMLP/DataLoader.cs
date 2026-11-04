using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace SonarMLP
{
    //each row of Features is one sonar reading (60 numbers), and Labels[i] is the answer for that row (1 = mine, 0 = rock).
    public class Dataset
    {
        public double[][] Features { get; }
        public double[] Labels { get; }
        public int Count => Labels.Length;

        public Dataset(double[][] features, double[] labels)
        {
            Features = features;
            Labels = labels;
        }
    }

    // Reads the csv file into a dataset.
    // File format: 60 comma-separated numbers, then a letter: 'M' (mine) or 'R' (rock).
    // The file has no header row.
    public static class DataLoader
    {
        public const int FeatureCount = 60;

        public static Dataset LoadSonarCsv(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Dataset file not found: {path}");

            var features = new List<double[]>();
            var labels = new List<double>();
            int skipped = 0;

            foreach (string rawLine in File.ReadLines(path))
            {
                string line = rawLine.Trim();
                if (line.Length == 0) continue; // ignore blank lines

                string[] parts = line.Split(',');

                // A valid row is exactly 60 numbers + 1 label. Anything else is skipped and counted.
                if (parts.Length != FeatureCount + 1)
                {
                    skipped++;
                    continue;
                }

                var row = new double[FeatureCount];
                bool rowOk = true;
                for (int i = 0; i < FeatureCount; i++)
                {
                    // InvariantCulture makes "0.0200" parse the same on every PC, regardless of regional settings (some use commas as decimals).
                    if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out row[i]))
                    {
                        rowOk = false;
                        break;
                    }
                }

                string label = parts[FeatureCount].Trim().ToUpperInvariant();
                if (!rowOk || (label != "M" && label != "R"))
                {
                    skipped++;
                    continue;
                }

                features.Add(row);
                labels.Add(label == "M" ? 1.0 : 0.0); // M (mine) = 1, R (rock) = 0
            }

            Console.WriteLine($"      -> Loaded {features.Count} rows ({skipped} skipped as invalid)");
            return new Dataset(features.ToArray(), labels.ToArray());
        }
    }
}