using System;
using System.Collections.Generic;

namespace SonarMLP
{
    // Min-max scaling: rescales each feature (column) to the range [0, 1] using   scaled = (value - min) / (max - min)
    public class MinMaxScaler
    {
        private double[] _min;
        private double[] _max;

        // Learn the min and max of every column from the given data.
        public void Fit(double[][] data)
        {
            int columns = data[0].Length;
            _min = new double[columns];
            _max = new double[columns];

            for (int c = 0; c < columns; c++)
            {
                _min[c] = double.MaxValue;
                _max[c] = double.MinValue;
                foreach (double[] row in data)
                {
                    if (row[c] < _min[c]) _min[c] = row[c];
                    if (row[c] > _max[c]) _max[c] = row[c];
                }
            }
        }

        // Apply the learned min/max to any data (train OR test). Returns a new copy.
        public double[][] Transform(double[][] data)
        {
            if (_min == null)
                throw new InvalidOperationException("Call Fit() before Transform().");

            var result = new double[data.Length][];
            for (int r = 0; r < data.Length; r++)
            {
                result[r] = new double[data[r].Length];
                for (int c = 0; c < data[r].Length; c++)
                {
                    double range = _max[c] - _min[c];
                    // If a column never changes, range is 0 -> avoid dividing by zero.
                    result[r][c] = range == 0 ? 0.0 : (data[r][c] - _min[c]) / range;
                }
            }
            return result;
        }
    }

    public static class DataSplitter
    {
        // Splits a dataset into train and test sets, keeping the mine/rock ratio the same in both
        // The fixed seed makes the split identical every run, so results are reproducible.
        public static (Dataset Train, Dataset Test) StratifiedSplit(
            Dataset data, double testRatio = 0.2, int seed = 42)
        {
            var rng = new Random(seed);

            // Collect the row numbers belonging to each class.
            var rockIdx = new List<int>();
            var mineIdx = new List<int>();
            for (int i = 0; i < data.Count; i++)
            {
                if (data.Labels[i] == 1.0) mineIdx.Add(i);
                else rockIdx.Add(i);
            }

            var trainIdx = new List<int>();
            var testIdx = new List<int>();

            // Split each class separately, then combine.
            foreach (var classIdx in new[] { rockIdx, mineIdx })
            {
                Shuffle(classIdx, rng);
                int testCount = (int)Math.Round(classIdx.Count * testRatio);
                for (int k = 0; k < classIdx.Count; k++)
                {
                    if (k < testCount) testIdx.Add(classIdx[k]);
                    else trainIdx.Add(classIdx[k]);
                }
            }

            // Shuffle so train/test aren't ordered "all rocks, then all mines".
            Shuffle(trainIdx, rng);
            Shuffle(testIdx, rng);

            return (Subset(data, trainIdx), Subset(data, testIdx));
        }

        // Fisher-Yates shuffle: walk backwards, swapping each item with a random earlier one.
        public static void Shuffle(List<int> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        private static Dataset Subset(Dataset data, List<int> indices)
        {
            var features = new double[indices.Count][];
            var labels = new double[indices.Count];
            for (int k = 0; k < indices.Count; k++)
            {
                features[k] = data.Features[indices[k]];
                labels[k] = data.Labels[indices[k]];
            }
            return new Dataset(features, labels);
        }
    }
}