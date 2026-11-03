using System;

namespace SonarMLP
{
    // Stores the four counts and computes metrics from them
    // Positive class = Mine (1), negative class = Rock (0)
    public class ConfusionMatrix
    {
        public int TruePositives { get; set; }   // mine predicted as mine
        public int TrueNegatives { get; set; }   // rock predicted as rock
        public int FalsePositives { get; set; }  // rock predicted as mine
        public int FalseNegatives { get; set; }  // mine predicted as rock

        public int Total => TruePositives + TrueNegatives + FalsePositives + FalseNegatives;

        public double Accuracy =>
            Total == 0 ? 0 : (double)(TruePositives + TrueNegatives) / Total;

        // checks to avoid dividing by zero
        public double Precision =>
            (TruePositives + FalsePositives) == 0 ? 0 : (double)TruePositives / (TruePositives + FalsePositives);

        public double Recall =>
            (TruePositives + FalseNegatives) == 0 ? 0 : (double)TruePositives / (TruePositives + FalseNegatives);

        public double F1 =>
            (Precision + Recall) == 0 ? 0 : 2 * Precision * Recall / (Precision + Recall);
    }

    public static class Evaluator
    {
        // Runs every sample through the network and counts the four outcomes
        // Only reads the network, it never changes any weights
        public static ConfusionMatrix Evaluate(NeuralEngine network, Dataset data)
        {
            var matrix = new ConfusionMatrix();

            for (int i = 0; i < data.Count; i++)
            {
                int predicted = network.Predict(data.Features[i]).PredictedClass;
                int actual = (int)data.Labels[i];

                if (actual == 1 && predicted == 1) matrix.TruePositives++;
                else if (actual == 0 && predicted == 0) matrix.TrueNegatives++;
                else if (actual == 0 && predicted == 1) matrix.FalsePositives++;
                else matrix.FalseNegatives++;
            }

            return matrix;
        }

        // Prints the confusion matrix and all metrics for one dataset
        public static void Print(string name, ConfusionMatrix m)
        {
            Console.WriteLine($"\n=== {name} RESULTS ({m.Total} samples) ===");
            Console.WriteLine("                  Predicted Rock   Predicted Mine");
            Console.WriteLine($"Actual Rock       {m.TrueNegatives,14}   {m.FalsePositives,14}");
            Console.WriteLine($"Actual Mine       {m.FalseNegatives,14}   {m.TruePositives,14}");
            Console.WriteLine($"Accuracy:  {m.Accuracy:P2}");
            Console.WriteLine($"Precision: {m.Precision:P2}");
            Console.WriteLine($"Recall:    {m.Recall:P2}");
            Console.WriteLine($"F1 Score:  {m.F1:P2}");
        }
    }
}