using System;
using System.IO;

namespace SonarMLP
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("===============================================================");
            Console.WriteLine("   ACOUSTIC SONAR CLASSIFIER: MINES VS ROCKS (3-LAYER MLP)     ");
            Console.WriteLine("===============================================================\n");

            // 1. DATA LOADING & INGESTION
            string dataPath = "sonar.all-data.csv";
            if (!File.Exists(dataPath))
            {
                // Fallback check if the user kept the original UCI file name without .csv
                if (File.Exists("sonar.all-data"))
                {
                    dataPath = "sonar.all-data";
                }
                else
                {
                    Console.WriteLine($"[!] Error: Dataset '{dataPath}' not found.");
                    Console.WriteLine("    Make sure the dataset is placed in the project root directory.");
                    return;
                }
            }

            Console.WriteLine($"[1/4] Loading dataset from: {dataPath}");
            var rawRecords = DataPipeline.LoadCsv(dataPath);
            Console.WriteLine($"      -> Successfully loaded {rawRecords.Count} samples (60 features each).");

            // 2. PREPROCESSING: MIN-MAX SCALING & TRAIN/TEST SPLIT
            Console.WriteLine("\n[2/4] Preprocessing dataset...");
            DataPipeline.MinMaxScale(rawRecords, 60);
            Console.WriteLine("      -> Min-Max normalization completed across all 60 frequency attributes.");

            var (trainSet, testSet) = DataPipeline.SplitTrainTest(rawRecords, trainRatio: 0.8, seed: 101);
            Console.WriteLine($"      -> Dataset split (80/20): {trainSet.Count} training samples, {testSet.Count} test samples.");

            // 3. HYPERPARAMETER CONFIGURATION & MODEL INITIALIZATION
            int inputNodes = 60;
            int hiddenNodes = 24;      // Balanced capacity to prevent underfitting & memorization
            int outputNodes = 1;       // Single Sigmoid unit for binary classification
            double learningRate = 0.15;
            int epochs = 1000;

            Console.WriteLine("\n[3/4] Initializing 3-Layer Neural Engine...");
            Console.WriteLine($"      -> Topology: [{inputNodes} Inputs] -> [{hiddenNodes} Hidden Nodes] -> [{outputNodes} Output Node]");
            Console.WriteLine($"      -> Hyperparameters: Learning Rate (eta) = {learningRate}, Epochs = {epochs}");

            var network = new NeuralEngine(inputNodes, hiddenNodes, outputNodes, learningRate, seed: 42);

            // 4. TRAINING LOOP & CONVERGENCE LOGGING (DELIVERABLE 2)
            string logPath = "training_log.csv";
            using var logWriter = new StreamWriter(logPath);
            logWriter.WriteLine("Epoch,AverageLoss");

            Console.WriteLine("\n===============================================================");
            Console.WriteLine("                      TRAINING PROGRESS                        ");
            Console.WriteLine("===============================================================");
            Console.WriteLine("Epoch\t\tAverage Loss\t\tStatus");
            Console.WriteLine("---------------------------------------------------------------");

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double totalEpochLoss = 0.0;

                // Stochastic / Online sample iteration
                foreach (var sample in trainSet)
                {
                    totalEpochLoss += network.TrainSample(sample.Features, sample.Target);
                }

                double avgLoss = totalEpochLoss / trainSet.Count;

                // Log every 100 epochs (and the very first epoch)
                if (epoch % 100 == 0 || epoch == 1)
                {
                    Console.WriteLine($"{epoch}\t\t{avgLoss:F6}\t\tConverging");
                    logWriter.WriteLine($"{epoch},{avgLoss:F6}");
                }
            }

            Console.WriteLine("---------------------------------------------------------------");
            Console.WriteLine($"[+] Training complete. Loss history exported to '{logPath}'.\n");

            // 5. TEST SET EVALUATION & PERFORMANCE ANALYSIS (DELIVERABLE 4)
            Console.WriteLine("===============================================================");
            Console.WriteLine("                 TEST SET PERFORMANCE ANALYSIS                 ");
            Console.WriteLine("===============================================================");
            
            var evalResult = Evaluator.Evaluate(network, testSet);
            evalResult.Display();

            Console.WriteLine("===============================================================\n");
        }
    }
}