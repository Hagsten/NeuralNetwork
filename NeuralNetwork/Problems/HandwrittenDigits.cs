using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Problems
{
    public static class HandwrittenDigits
    {
        public static void Run()
        {
            Console.WriteLine("Creating neural network...");

            var network = new NeuralNetwork(784, 100, 10, 0.3);

            var dataset = File.ReadAllLines(@"C:/Temp/mnist_train.csv");

            var allInputs = dataset
                .Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            Console.WriteLine($"Training network with {allInputs.Length} samples...");

            var normalizedInputs = allInputs.Select(x => new
            {
                Answer = x[0],
                Inputs = NormalizeInput(x.Skip(1).ToArray())
            }).ToArray();

            var s = new Stopwatch();
            s.Start();

            foreach (var input in normalizedInputs)
            {
                var targets = Enumerable.Range(0, 10).Select(x => 0.0).Select(x => x + 0.01).ToArray();
                targets[int.Parse(input.Answer)] = 0.99;

                network.Train(input.Inputs, targets);
            }

            s.Stop();
            Console.WriteLine($"Training complete in {s.ElapsedMilliseconds}ms{Environment.NewLine}");
            Console.WriteLine("Querying network...");

            var queryDataset = File.ReadAllLines(@"C:/Temp/mnist_test.csv"); ;

            var queryInputs = queryDataset
                .Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            var scoreCard = new List<bool>();

            foreach (var input in queryInputs)
            {
                var normalizedInput = NormalizeInput(input.Skip(1).ToArray());
                var correctAnswer = int.Parse(input[0]);
                var response = network.Query(normalizedInput);

                var max = response.Max(x => x);
                var f = response.ToList().IndexOf(max);

                scoreCard.Add(correctAnswer == f);
            }

            Console.WriteLine($"Performed {scoreCard.Count} queries. Correct answers were {scoreCard.Count(x => x)}.");
            Console.WriteLine($"Network has a performance of {scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)}");
        }

        private static double[] NormalizeInput(string[] input)
        {
            return input
                .Select(double.Parse)
                .Select(y => (y / 255) * 0.99 + 0.01)
                .ToArray();
        }
    }
}
