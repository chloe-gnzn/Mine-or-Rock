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
    }
}