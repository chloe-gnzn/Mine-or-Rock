using System;

namespace SonarMLP
{
    //3-layer multi-layer perceptron (input, hidden, output) for binary classification
    public class NeuralEngine
    {
        //network dimensions and hyperparameters
        public int InputDim { get; }      //60 inputs for sonar frequencies
        public int HiddenDim { get; }     //24 units for non-linear feature mapping
        public int OutputDim { get; }     //1 output unit for binary probability
        public double LearningRate { get; set; } //eta step size for weight updates

        //weight matrices and bias vectors
        public double[,] WeightsIH { get; private set; } //inputdim to hiddendim weights
        public double[] BiasesH { get; private set; }     //hiddendim biases
        public double[,] WeightsHO { get; private set; } //hiddendim to outputdim weights
        public double[] BiasesO { get; private set; }     //outputdim biases

        private readonly Random _rng;

        //initialize dimensions, allocate arrays, and trigger weight initialization
        public NeuralEngine(int inputDim = 60, int hiddenDim = 24, int outputDim = 1, double lr = 0.15, int seed = 42)
        {
            InputDim = inputDim;
            HiddenDim = hiddenDim;
            OutputDim = outputDim;
            LearningRate = lr;
            _rng = new Random(seed);

            //allocate layer weight and bias arrays
            WeightsIH = new double[InputDim, HiddenDim];
            BiasesH = new double[HiddenDim];
            WeightsHO = new double[HiddenDim, OutputDim];
            BiasesO = new double[OutputDim];

            InitializeWeights();
        }

        //xavier uniform initialization to keep variance stable and prevent vanishing gradients
        private void InitializeWeights()
        {
            //input to hidden layer weight initialization: uniform limit = sqrt(6 / (fan_in + fan_out))
            double limitIH = Math.Sqrt(6.0 / (InputDim + HiddenDim));
            for (int i = 0; i < InputDim; i++)
            {
                for (int h = 0; h < HiddenDim; h++)
                {
                    WeightsIH[i, h] = (_rng.NextDouble() * 2 * limitIH) - limitIH;
                }
            }

            //hidden to output layer weight initialization and zero-bias setup
            double limitHO = Math.Sqrt(6.0 / (HiddenDim + OutputDim));
            for (int h = 0; h < HiddenDim; h++)
            {
                BiasesH[h] = 0.0;
                for (int o = 0; o < OutputDim; o++)
                {
                    WeightsHO[h, o] = (_rng.NextDouble() * 2 * limitHO) - limitHO;
                }
            }

            //output layer zero-bias setup
            for (int o = 0; o < OutputDim; o++)
            {
                BiasesO[o] = 0.0;
            }
        }

        //sigmoid activation with value clamping to prevent numerical overflow in math.exp
        private static double Sigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        //derivative of sigmoid: f'(output) = output * (1.0 - output)
        private static double SigmoidDerivative(double output)
        {
            return output * (1.0 - output);
        }

        //forward pass through input, hidden, and output layers
        public double[] ComputeForward(double[] inputs, out double[] hiddenActivations)
        {
            hiddenActivations = new double[HiddenDim];
            double[] outputs = new double[OutputDim];

            //input to hidden forward step
            for (int h = 0; h < HiddenDim; h++)
            {
                double sum = BiasesH[h];
                for (int i = 0; i < InputDim; i++)
                {
                    sum += inputs[i] * WeightsIH[i, h];
                }
                hiddenActivations[h] = Sigmoid(sum);
            }

            //hidden to output forward step
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

        //executes forward pass, computes sample mse loss, and backpropagates weight updates
        public double TrainSample(double[] inputs, double target)
        {
            //forward pass
            double[] output = ComputeForward(inputs, out double[] hiddenActivations);
            double yHat = output[0];

            //sample mse loss: 0.5 * (target - yHat)^2
            double error = target - yHat;
            double sampleLoss = 0.5 * Math.Pow(error, 2);

            //output layer delta: delta_o = (target - yHat) * f'(yHat)
            double deltaO = error * SigmoidDerivative(yHat);

            //hidden layer delta: delta_h = (delta_o * W_ho) * f'(hidden)
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

        //binary classification thresholding at 0.5
        public (int PredictedClass, double Confidence) Predict(double[] inputs)
        {
            double probability = ComputeForward(inputs, out _)[0];
            int classification = probability >= 0.5 ? 1 : 0; //1 for mine, 0 for rock
            return (classification, probability);
        }
    }
}