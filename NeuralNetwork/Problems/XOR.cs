using System;

namespace NeuralNetwork.Problems
{
    public static class XOR
    {
        public static void Run()
        {
            Console.WriteLine("Creating neural network...");

            var network = new NeuralNetwork(2, 5, 1, 1.8);

            for (int epoch = 0; epoch < 2000; epoch++)
            {
                network.Train(new[] { 0.01, 0.01 }, new[] { 0.01 });
                network.Train(new[] { 0.01, 0.99 }, new[] { 0.99 });
                network.Train(new[] { 0.99, 0.01 }, new[] { 0.99 });
                network.Train(new[] { 0.99, 0.99 }, new[] { 0.01 });
            }

            var true_1 = network.Query(new[] { 0.99, 0.01 });
            var true_2 = network.Query(new[] { 0.01, 0.99 });
            var false_1 = network.Query(new[] { 0.01, 0.01 });
            var false_2 = network.Query(new[] { 0.99, 0.99 });

            Console.WriteLine($"Networks answer for true_1: {true_1[0]} which is {(true_1[0] > 0.5)}");
            Console.WriteLine($"Networks answer for true_2: {true_2[0]} which is {(true_2[0] > 0.5)}");
            Console.WriteLine($"Networks answer for false_1: {false_1[0]} which is {(false_1[0] > 0.5)}");
            Console.WriteLine($"Networks answer for false_2: {false_2[0]} which is {(false_2[0] > 0.5)}");
        }
    }
}
