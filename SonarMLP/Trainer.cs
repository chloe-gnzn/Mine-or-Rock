using System;
using System.Collections.Generic;
using System.IO;

namespace SonarMLP
{
    // Runs the training loop and measures how the network is doing
    public class Trainer
    {
        private readonly NeuralEngine _network;

        public Trainer(NeuralEngine network)
        {
            _network = network;
        }

        // Trains for a number of epochs. One epoch = one full pass over the training set.
        // Every 'logEvery' epochs, it prints and saves loss and accuracy for train and test
        public void Train(Dataset train, Dataset test, int epochs, int logEvery, string logPath, int seed = 42)
        {
            var rng = new Random(seed);

            // A list of row numbers 0..N-1 that reshuffles every epoch
            var order = new List<int>();
            for (int i = 0; i < train.Count; i++) order.Add(i);

            using var writer = new StreamWriter(logPath);
            writer.WriteLine("Epoch,TrainLoss,TestLoss,TrainAccuracy,TestAccuracy");

            Console.WriteLine("Epoch\tTrainLoss\tTestLoss\tTrainAcc\tTestAcc");
            Console.WriteLine("-----------------------------------------------------------");

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                // Different sample order each epoch stops the network from memorizing the order of the data
                DataSplitter.Shuffle(order, rng);

                foreach (int idx in order)
                {
                    _network.TrainSample(train.Features[idx], train.Labels[idx]);
                }

                if (epoch == 1 || epoch % logEvery == 0)
                {
                    double trainLoss = ComputeLoss(train);
                    double testLoss = ComputeLoss(test);
                    double trainAcc = ComputeAccuracy(train);
                    double testAcc = ComputeAccuracy(test);

                    Console.WriteLine($"{epoch}\t{trainLoss:F6}\t{testLoss:F6}\t{trainAcc:P1}\t{testAcc:P1}");
                    writer.WriteLine($"{epoch},{trainLoss:F6},{testLoss:F6},{trainAcc:F4},{testAcc:F4}");
                }
            }
        }

        // Average loss over a dataset, using the same formula as the engine: 0.5 * (target - output)^2
        // This only measures and does not change any weights
        public double ComputeLoss(Dataset data)
        {
            double total = 0;
            for (int i = 0; i < data.Count; i++)
            {
                double output = _network.Predict(data.Features[i]).Confidence;
                double error = data.Labels[i] - output;
                total += 0.5 * error * error;
            }
            return total / data.Count;
        }

        // Fraction of samples classified correctly (0.0 to 1.0)
        public double ComputeAccuracy(Dataset data)
        {
            int correct = 0;
            for (int i = 0; i < data.Count; i++)
            {
                int predicted = _network.Predict(data.Features[i]).PredictedClass;
                if (predicted == (int)data.Labels[i]) correct++;
            }
            return (double)correct / data.Count;
        }
    }
}