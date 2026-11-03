using System;

namespace SonarMLP
{
    public class NeuralEngine
    {
        public int InputDim { get; }
        public int HiddenDim { get; }
        public int OutputDim { get; }
        public double LearningRate { get; set; }

        public double[,] WeightsIH { get; private set; } //inputdim, hiddendim
        public double[] BiasesH { get; private set; }     //hiddendim
        public double[,] WeightsHO { get; private set; } //hiddendim, outputdim
        public double[] BiasesO { get; private set; }     //outputdim

        private readonly Random _rng;

        public NeuralEngine(int inputDim = 60, int hiddenDim = 24, int outputDim = 1, double lr = 0.15, int seed = 42)
        {
            InputDim = inputDim;
            HiddenDim = hiddenDim;
            OutputDim = outputDim;
            LearningRate = lr;
            _rng = new Random(seed);

            WeightsIH = new double[InputDim, HiddenDim];
            BiasesH = new double[HiddenDim];
            WeightsHO = new double[HiddenDim, OutputDim];
            BiasesO = new double[OutputDim];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            double limitIH = Math.Sqrt(6.0 / (InputDim + HiddenDim));
            for (int i = 0; i < InputDim; i++)
            {
                for (int h = 0; h < HiddenDim; h++)
                {
                    WeightsIH[i, h] = (_rng.NextDouble() * 2 * limitIH) - limitIH;
                }
            }

            double limitHO = Math.Sqrt(6.0 / (HiddenDim + OutputDim));
            for (int h = 0; h < HiddenDim; h++)
            {
                BiasesH[h] = 0.0;
                for (int o = 0; o < OutputDim; o++)
                {
                    WeightsHO[h, o] = (_rng.NextDouble() * 2 * limitHO) - limitHO;
                }
            }

            for (int o = 0; o < OutputDim; o++)
            {
                BiasesO[o] = 0.0;
            }
        }

        private static double Sigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double SigmoidDerivative(double output)
        {
            return output * (1.0 - output);
        }

        public double[] ComputeForward(double[] inputs, out double[] hiddenActivations)
        {
            hiddenActivations = new double[HiddenDim];
            double[] outputs = new double[OutputDim];

            //input to hidden
            for (int h = 0; h < HiddenDim; h++)
            {
                double sum = BiasesH[h];
                for (int i = 0; i < InputDim; i++)
                {
                    sum += inputs[i] * WeightsIH[i, h];
                }
                hiddenActivations[h] = Sigmoid(sum);
            }

            //hidden to output
            for (int o = 0; o < OutputDim; o++)
            {
                double sum = BiasesO[o];
                for (int h = 0; h < HiddenDim; h++)
                {
                    sum += hiddenActivations[h] * WeightsHO[h, o];
                }
                outputs[o] = Sigmoid(sum);
            }

            return outputs;
        }

        public double TrainSample(double[] inputs, double target)
        {
            //forward pass
            double[] output = ComputeForward(inputs, out double[] hiddenActivations);
            double yHat = output[0];

            //sample loss
            double error = target - yHat;
            double sampleLoss = 0.5 * Math.Pow(error, 2);

            //output layer delta
            double deltaO = error * SigmoidDerivative(yHat);

            //hidden layer datas
            double[] deltasH = new double[HiddenDim];
            for (int h = 0; h < HiddenDim; h++)
            {
                deltasH[h] = (deltaO * WeightsHO[h, 0]) * SigmoidDerivative(hiddenActivations[h]);
            }

            //update hidden to output weights and biases
            for (int h = 0; h < HiddenDim; h++)
            {
                WeightsHO[h, 0] += LearningRate * deltaO * hiddenActivations[h];
            }
            BiasesO[0] += LearningRate * deltaO;

            //update input to hidden weights and biases
            for (int h = 0; h < HiddenDim; h++)
            {
                for (int i = 0; i < InputDim; i++)
                {
                    WeightsIH[i, h] += LearningRate * deltasH[h] * inputs[i];
                }
                BiasesH[h] += LearningRate * deltasH[h];
            }

            return sampleLoss;
        }

        public (int PredictedClass, double Confidence) Predict(double[] inputs)
        {
            double probability = ComputeForward(inputs, out _)[0];
            int classification = probability >= 0.5 ? 1 : 0; //1 for mine, 0 for rocks
            return (classification, probability);
        }
    }
}