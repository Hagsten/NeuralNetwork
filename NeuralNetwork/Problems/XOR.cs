using System;

namespace NeuralNetwork.Problems
{
    public static class XOR
    {
        public static void Run()
        {
            Console.WriteLine("Creating neural network...");

            var network = new NeuralNetwork(2, 2, 1, 1.8);

            for (int epoch = 0; epoch < 2000; epoch++)
            {
                network.Train(new[] { 0.01, 0.01 }, new[] { 0.01 });
                network.Train(new[] { 0.01, 0.99 }, new[] { 0.99 });
                network.Train(new[] { 0.99, 0.01 }, new[] { 0.99 });
                network.Train(new[] { 0.99, 0.99 }, new[] { 0.01 });
            }

            var result = network.Query(new[] { 0.99, 0.01 });

            Console.WriteLine($"Networks best answerer: {result[0]}");
        }
    }
}
