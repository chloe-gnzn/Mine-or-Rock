using System;

namespace SonarMLP
{
    //file for the 3-layer multi-layer perceptron

    public class NeuralEngine
    {
        public int InputDim //no. of nodes in the input laye
        {
            get;
        }
        public int HiddenDim //no. of processing neurons
        {
            get;
        }
        public int OutputDim //no. of output unit
        {
            get;
        }
        public double LearningRate//size scaling factor applied to weight updates
        {
            get; set;
        }

        //matrices
        public double[,] WeightsIH //input dim, hidden dim
        {
            get; private set;
        }
        public double[,] BiasesH //hidden dim
        {
            get; private set;
        }
        public double[,] WeightsHO //. hidden dim, output dim
        {
            get; private set;
        }
        public double[,] BiasesO //output dim
        {
            get; private set;
        }

        private readonly Random _rng;

        //constructor 
        public NeuralEngine(int inputDim = 60, int hiddenDim = 24, int outputDim = 1, double lr = 0.15, int seed = 42)
        {
            InputDim = inputDim;
            HiddenDim = hiddenDim;
            OutputDim = outputDim;
            LearningRate = lr;
            _rng = new Random(seed);

            WeightsHO = new double[HiddenDim, OutputDim];
            WeightsIH = new double[InputDim, HiddenDim]; 
            BiasesH = new double[HiddenDim];
            BiasesO = new double[OutputDim];

            InitializeWeights();
        }

        private void InitializeWeig()//weight initialization logic
        {
            double limitIH = Math.Sqrt(96.0 / (InputDim + HiddenDim));
            for(int i = 0; i< InputDim; i++)
            {
                for(int h = 0; h < HiddenDim; h++)
                {
                    WeightsIH[i,h] = (_rng.NextDouble() * 2 * limitIH) - limitIH;
                }
            }

            double limitHO = Math.Sqrt(6.0 / (HiddenDim + OutputDim));
            for(int h = 0; h < HiddenDim; h++)
            {
                BiasesH[h] = 0.0;
                for(int o = 0; o < OutputDim; o++)
                {
                    WeightsHO[h,o] = (_rng.NextDouble() * 2 * limitHO) - limitHO;
                }
            }

            for(int o = 0; o < OutputDim ; o++)
            {
                BiasesO[o] = 0.0;
            }
        }

        //sigmoid activation 
        private static double Sigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        //f'(y) = y * (1 - y)
        private static double SigmoidDerivative(double output){
            output * (1.0 - output);
        }

        //compute the forward pass across input -> hidden -> output layers
        public double[] ComputeForward(double[] inputs, out double[] hiddenActivations)
        {
            hiddenActivations = new double[HiddenDim];
            double[] outputs = new double[OutputDim];

            //input layer to hidden layer
            for (int h = 0; h < HiddenDim; h++)
            {
                double sum = BiasesH[h];
                for (int i = 0; i < InputDim; i++)
                {
                    sum += inputs[i] * WeightsIH[i, h];
                }
                hiddenActivations[h] = Sigmoid(sum);
            }

            //hidden layer to output layer
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

            //E = 0.5 * (target - yHat)^2
            double error = target - yHat;
            double sampleLoss = 0.5 * Math.Pow(error, 2);

            //delta_o = (target - yHat) * f'(yHat)
            double deltaO = error * SigmoidDerivative(yHat);

            //delta_h = (delta_o * W_ho) * f'(a_h)
            double[] deltasH = new double[HiddenDim];
            for (int h = 0; h < HiddenDim; h++)
            {
                deltasH[h] = (deltaO * WeightsHO[h, 0]) * SigmoidDerivative(hiddenActivations[h]);
            }

            //update hidden
            for (int h = 0; h < HiddenDim; h++)
            {
                WeightsHO[h, 0] += LearningRate * deltaO * hiddenActivations[h];
            }
            BiasesO[0] += LearningRate * deltaO;

            //update input
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
        //inference method predicting binary class
        public (int PredictedClass, double Confidence) Predict(double[] inputs)
        {
            double probability = ComputeForward(inputs, out _)[0];
            int classification = probability >= 0.5 ? 1 : 0; //1 if mine, 0 if rock
            return (classification, probability);
        }
    }
}