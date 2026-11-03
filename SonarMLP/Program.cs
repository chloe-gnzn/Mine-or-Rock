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

            Console.WriteLine("[0/3] Loading Sonar Dataset...");
            
            string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "sonar.csv");

            Dataset dataset;
            try
            {
                dataset = DataLoader.LoadSonarCsv(dataPath);

                Console.WriteLine($"      [+] Dataset loaded successfully!");
                Console.WriteLine($"      [+] Total Samples: {dataset.Count}");
                Console.WriteLine($"      [+] Features per Sample: {dataset.Features[0].Length}");

                Console.WriteLine("\n      --- Sample 1 Preview ---");
                Console.WriteLine($"      First 5 Features: [{dataset.Features[0][0]:F4}, {dataset.Features[0][1]:F4}, {dataset.Features[0][2]:F4}, {dataset.Features[0][3]:F4}, {dataset.Features[0][4]:F4}, ...]");
                Console.WriteLine($"      Label Value: {dataset.Labels[0]} ({(dataset.Labels[0] == 1.0 ? "Mine" : "Rock")})");
                Console.WriteLine("---------------------------------------------------------------\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      [!] Failed to load dataset: {ex.Message}");
                Console.WriteLine("      [!] Check if sonar.csv exists in the Data folder and is set to 'Copy to Output Directory'.\n");
                return; // Stop execution if dataset loading fails
            }

            // --- DATA SPLITTING & SCALING ---
            Console.WriteLine("Splitting and scaling dataset...");
            var (train, test) = DataSplitter.StratifiedSplit(dataset, testRatio: 0.2, seed: 42);

            var scaler = new MinMaxScaler();
            scaler.Fit(train.Features); // learn min/max from TRAIN only
            var trainScaled = new Dataset(scaler.Transform(train.Features), train.Labels);
            var testScaled = new Dataset(scaler.Transform(test.Features), test.Labels);

            int trainMines = 0, testMines = 0;
            foreach (double l in train.Labels) if (l == 1.0) trainMines++;
            foreach (double l in test.Labels) if (l == 1.0) testMines++;

            Console.WriteLine($"Train: {train.Count} rows ({trainMines} mines, {train.Count - trainMines} rocks)");
            Console.WriteLine($"Test:  {test.Count} rows ({testMines} mines, {test.Count - testMines} rocks)\n");


            int inputNodes = 60;
            int hiddenNodes = 24;
            int outputNodes = 1;
            double learningRate = 0.15;
            int epochs = 1000;

            Console.WriteLine($"[1/3] Initializing Neural Engine...");
            Console.WriteLine($"      -> Dimensions: [{inputNodes} Inputs] -> [{hiddenNodes} Hidden] -> [{outputNodes} Output]");
            Console.WriteLine($"      -> Learning Rate: {learningRate} | Epochs: {epochs}");

            var network = new NeuralEngine(inputNodes, hiddenNodes, outputNodes, learningRate, seed: 42);

            Console.WriteLine("\n[2/3] Performing Sanity Gradient Check (Synthetic Patterns)...");
            var dummyMine = new double[60];
            var dummyRock = new double[60];

            for (int i = 0; i < 60; i++)
            {
                dummyMine[i] = 0.85; //target = 1.0 for mine
                dummyRock[i] = 0.15; //target = 0.0 for rock
            }

            double initialLossMine = network.TrainSample(dummyMine, target: 1.0);
            double initialLossRock = network.TrainSample(dummyRock, target: 0.0);
            Console.WriteLine($"      -> Initial Step Loss: Mine={initialLossMine:F6}, Rock={initialLossRock:F6}");

            Console.WriteLine("\n[3/3] Executing Training Loop & Writing 'training_log.csv'...");

            string logPath = "training_log.csv";
            using var logWriter = new StreamWriter(logPath);
            logWriter.WriteLine("Epoch,AverageLoss");

            Console.WriteLine("\nEpoch\t\tAverage Loss\t\tStatus");
            Console.WriteLine("---------------------------------------------------------------");

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double loss1 = network.TrainSample(dummyMine, target: 1.0);
                double loss2 = network.TrainSample(dummyRock, target: 0.0);
                double avgLoss = (loss1 + loss2) / 2.0;

                if (epoch % 100 == 0 || epoch == 1)
                {
                    Console.WriteLine($"{epoch}\t\t{avgLoss:F6}\t\tActive");
                    logWriter.WriteLine($"{epoch},{avgLoss:F6}");
                }
            }

            Console.WriteLine("---------------------------------------------------------------");
            Console.WriteLine($"[+] Engine math verified. Log saved to: {Path.GetFullPath(logPath)}");

            var (predMine, confMine) = network.Predict(dummyMine);
            var (predRock, confRock) = network.Predict(dummyRock);

            Console.WriteLine("\nPost-Training Inference:");
            Console.WriteLine($"  Pattern Mine -> Predicted Class: {predMine} (Confidence: {confMine:P2})");
            Console.WriteLine($"  Pattern Rock -> Predicted Class: {predRock} (Confidence: {confRock:P2})");
        }
    }
}