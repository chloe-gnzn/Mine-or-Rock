using System;
using System.IO;

namespace SonarMLP
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("===============================================================");
            Console.WriteLine("         3-LAYER BACKPROPAGATION MLP w/ BINARY CLASSIFICATION    ");
            Console.WriteLine("===============================================================\n");

            // ---------- [0/3] LOAD DATA ----------
            Console.WriteLine("[0/3] Loading Sonar Dataset...");

            string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "sonar.csv");

            Dataset dataset;
            try
            {
                dataset = DataLoader.LoadSonarCsv(dataPath);

                Console.WriteLine($"      [+] Dataset loaded successfully!");
                Console.WriteLine($"      [+] Total Samples: {dataset.Count}");
                Console.WriteLine($"      [+] Features per Sample: {dataset.Features[0].Length}");
                Console.WriteLine("---------------------------------------------------------------\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      [!] Failed to load dataset: {ex.Message}");
                Console.WriteLine("      [!] Check if sonar.csv exists in the Data folder and is set to 'Copy to Output Directory'.\n");
                return; // Stop execution if dataset loading fails
            }

            // ---------- [1/3] SPLIT & SCALE ----------
            Console.WriteLine("[1/3] Splitting and scaling dataset...");
            var (train, test) = DataSplitter.StratifiedSplit(dataset, testRatio: 0.2, seed: 42);

            var scaler = new MinMaxScaler();
            scaler.Fit(train.Features); // learn min/max from TRAIN only (no leakage)
            var trainScaled = new Dataset(scaler.Transform(train.Features), train.Labels);
            var testScaled = new Dataset(scaler.Transform(test.Features), test.Labels);

            int trainMines = 0, testMines = 0;
            foreach (double l in train.Labels) if (l == 1.0) trainMines++;
            foreach (double l in test.Labels) if (l == 1.0) testMines++;

            Console.WriteLine($"      -> Train: {train.Count} rows ({trainMines} mines, {train.Count - trainMines} rocks)");
            Console.WriteLine($"      -> Test:  {test.Count} rows ({testMines} mines, {test.Count - testMines} rocks)\n");

            // ---------- [2/3] BUILD NETWORK ----------
            int inputNodes = 60;
            int hiddenNodes = 24;
            int outputNodes = 1;
            double learningRate = 0.15;
            int epochs = 1000;

            Console.WriteLine("[2/3] Initializing Neural Engine...");
            Console.WriteLine($"      -> Dimensions: [{inputNodes} Inputs] -> [{hiddenNodes} Hidden] -> [{outputNodes} Output]");
            Console.WriteLine($"      -> Learning Rate: {learningRate} | Epochs: {epochs}\n");

            var network = new NeuralEngine(inputNodes, hiddenNodes, outputNodes, learningRate, seed: 42);
            var trainer = new Trainer(network);

            // ---------- [3/3] TRAIN ----------
            Console.WriteLine("[3/3] Training on real Sonar data & writing 'training_log.csv'...\n");

            string logPath = "training_log.csv";
            trainer.Train(trainScaled, testScaled, epochs, logEvery: 100, logPath: logPath);

            Console.WriteLine("-----------------------------------------------------------");
            Console.WriteLine($"[+] Log saved to: {Path.GetFullPath(logPath)}");

            Console.WriteLine($"\nFinal Train Accuracy: {trainer.ComputeAccuracy(trainScaled):P2}");
            Console.WriteLine($"Final Test Accuracy:  {trainer.ComputeAccuracy(testScaled):P2}");

            Evaluator.Print("TRAIN", Evaluator.Evaluate(network, trainScaled));
            Evaluator.Print("TEST", Evaluator.Evaluate(network, testScaled));

            // ---------- HYPERPARAMETER SWEEP ----------
            Console.WriteLine("\n[4/4] Hyperparameter sweep ...");

            // Split the TRAINING data again: ~133 rows to train on, ~34 to validate on.
            var (fitSet, valSet) = DataSplitter.StratifiedSplit(trainScaled, testRatio: 0.2, seed: 7);

            var sweepResults = HyperparameterSweep.Run(
                fitSet, valSet,
                learningRates: new[] { 0.01, 0.05, 0.15, 0.3 },
                hiddenSizes: new[] { 8, 16, 24, 48 },
                checkpoints: new[] { 100, 300, 500, 1000 },
                seeds: new[] { 1, 2, 3 });

            HyperparameterSweep.WriteCsv(sweepResults, "hyperparameter_results.csv");
            HyperparameterSweep.PrintTop(sweepResults, 5);
        }
    }
}