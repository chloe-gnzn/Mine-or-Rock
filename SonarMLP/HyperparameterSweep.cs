using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SonarMLP
{
    // one row of results is a configuration at one epoch checkpoint, averaged over several random starting points
    public class SweepResult
    {
        public double LearningRate { get; set; }
        public int HiddenNodes { get; set; }
        public int Epochs { get; set; }
        public double TrainLoss { get; set; }
        public double ValLoss { get; set; }
        public double TrainAccuracy { get; set; }
        public double ValAccuracy { get; set; }
    }

    public static class HyperparameterSweep
    {
        // Tries every (learning rate, hidden size) pair. Each run trains up to the last checkpoint and records results at each checkpoint along the way, so epochs get tested for free.
        public static List<SweepResult> Run(
            Dataset train, Dataset validation,
            double[] learningRates, int[] hiddenSizes, int[] checkpoints, int[] seeds)
        {
            var results = new List<SweepResult>();
            int inputDim = train.Features[0].Length;
            int maxEpochs = checkpoints[checkpoints.Length - 1];

            foreach (double lr in learningRates)
            {
                foreach (int hidden in hiddenSizes)
                {
                    Console.WriteLine($"      Testing lr={lr}, hidden={hidden} ...");

                    // sums[c, 0..3] = running totals of trainLoss, valLoss, trainAcc, valAcc at checkpoint c
                    var sums = new double[checkpoints.Length, 4];

                    foreach (int seed in seeds)
                    {
                        var network = new NeuralEngine(inputDim, hidden, 1, lr, seed);
                        var trainer = new Trainer(network); // reused only for its loss/accuracy helpers
                        var rng = new Random(seed);

                        var order = new List<int>();
                        for (int i = 0; i < train.Count; i++) order.Add(i);

                        int next = 0; // which checkpoint we're waiting for
                        for (int epoch = 1; epoch <= maxEpochs; epoch++)
                        {
                            DataSplitter.Shuffle(order, rng);
                            foreach (int idx in order)
                                network.TrainSample(train.Features[idx], train.Labels[idx]);

                            if (epoch == checkpoints[next])
                            {
                                sums[next, 0] += trainer.ComputeLoss(train);
                                sums[next, 1] += trainer.ComputeLoss(validation);
                                sums[next, 2] += trainer.ComputeAccuracy(train);
                                sums[next, 3] += trainer.ComputeAccuracy(validation);
                                next++;
                            }
                        }
                    }

                    // Average over the seeds and store one result per checkpoint
                    for (int c = 0; c < checkpoints.Length; c++)
                    {
                        results.Add(new SweepResult
                        {
                            LearningRate = lr,
                            HiddenNodes = hidden,
                            Epochs = checkpoints[c],
                            TrainLoss = sums[c, 0] / seeds.Length,
                            ValLoss = sums[c, 1] / seeds.Length,
                            TrainAccuracy = sums[c, 2] / seeds.Length,
                            ValAccuracy = sums[c, 3] / seeds.Length
                        });
                    }
                }
            }
            return results;
        }

        public static void WriteCsv(List<SweepResult> results, string path)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("LearningRate,HiddenNodes,Epochs,TrainLoss,ValLoss,TrainAccuracy,ValAccuracy");
            foreach (var r in results)
            {
                writer.WriteLine($"{r.LearningRate},{r.HiddenNodes},{r.Epochs}," +
                                 $"{r.TrainLoss:F6},{r.ValLoss:F6},{r.TrainAccuracy:F4},{r.ValAccuracy:F4}");
            }
        }

        // Prints the best configurations, ranked by validation loss 
        // Loss is used instead of accuracy because accuracy on ~34 samples can be very jumpy
        public static void PrintTop(List<SweepResult> results, int count)
        {
            Console.WriteLine("\nTop configurations by validation loss:");
            Console.WriteLine("Rank\tLR\tHidden\tEpochs\tValLoss\t\tValAcc\tTrainAcc");
            int rank = 1;
            foreach (var r in results.OrderBy(x => x.ValLoss).Take(count))
            {
                Console.WriteLine($"{rank++}\t{r.LearningRate}\t{r.HiddenNodes}\t{r.Epochs}\t{r.ValLoss:F6}\t{r.ValAccuracy:P1}\t{r.TrainAccuracy:P1}");
            }
        }
    }
}