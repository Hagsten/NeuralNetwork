using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Creating neural network...");

            var network = new NeuralNetwork(784, 100, 10, 0.3);

            var dataset = File.ReadAllLines(@"C:/Temp/mnist_train.csv");

            var allInputs = dataset
                .Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            Console.WriteLine($"Training network with {allInputs.Length} samples...");

            var foo = allInputs.Select(x => new
            {
                Answer = x[0],
                Inputs = NormalizeInput(x.Skip(1).ToArray())
            }).ToArray();

            var s = new Stopwatch();
            s.Start();

            foreach (var input in foo)
            {
                var targets = Enumerable.Range(0, 10).Select(x => 0).Select(Convert.ToDouble).Select(x => x + 0.01).ToArray();
                targets[int.Parse(input.Answer)] = 0.99;

                network.Train(input.Inputs, targets);
            }

            s.Stop();
            Console.WriteLine("Training complete in " + s.ElapsedMilliseconds);
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
            Console.ReadLine();
        }

        private static void ApproximateSine()
        {
            var n = new NeuralNetwork(1, 15, 1, 0.01);

            var inputs = new List<double>();

            for (var x = -Math.PI; x < Math.PI; x += 0.005)
            {
                inputs.Add(x);
            }

            var random = new Random();

            var shuffledInputs = inputs.OrderBy(x => random.NextDouble()).ToArray();

            var trainData = shuffledInputs.Take(shuffledInputs.Length / 2).ToArray();
            var testData = shuffledInputs.Skip(shuffledInputs.Length / 2).ToArray();

            for (var epoch = 0; epoch < 100; epoch++)
            {
                foreach (var data in trainData)
                {
                    var normalizedInput = data / Math.PI; //normalized [-1, 1]
                    var actual = (Math.Sin(normalizedInput) + 1) / 2; //normalized between [0, 1]

                    n.Train(new[] { normalizedInput }, new[] { actual });
                }
            }

            foreach (var data in testData.OrderBy(x => x).ToArray())
            {
                var normalizedInput = data / Math.PI;
                var result = n.Query(new[] { normalizedInput });
                var errorMargin = ((result[0] * 2) - 1) - Math.Sin(normalizedInput);
                Console.WriteLine("Error: " + errorMargin);
                Console.WriteLine($"Answer: {(result[0] * 2) - 1}. Actual: {Math.Sin(normalizedInput)}");
            }
        }

        private static void XOR()
        {
            var n = new NeuralNetwork(2, 3, 4, 0.3);

            for (int i = 0; i < 1000; i++)
            {
                n.Train(new[] { 0.01, 0.01 }, new[] { 0.01, 0.01, 0.01, 0.01 });
                n.Train(new[] { 0.01, 0.99 }, new[] { 0.01, 0.99, 0.99, 0.01 });
                n.Train(new[] { 0.99, 0.01 }, new[] { 0.01, 0.99, 0.99, 0.01 });
                n.Train(new[] { 0.99, 0.99 }, new[] { 0.01, 0.01, 0.01, 0.01 });
            }

            var result = n.Query(new[] { 0.99, 0.01 });
        }

        private static void Iris()
        {
            var possibleResults = new List<string> { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

            var network = new NeuralNetwork(4, 2, 3, 0.2);

            var dataset = File.ReadAllLines(@"C:/Temp/iris.csv");

            var allInputs = dataset.Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            var shuffledInputs = Shuffle(allInputs);

            var trainDataSet = shuffledInputs.Take(100).ToArray();
            const int epochs = 200;

            Console.WriteLine($"Training network with {trainDataSet.Length} samples using {epochs} epochs...");

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var input in trainDataSet)
                {
                    var targets = new[] { 0.01, 0.01, 0.01 };
                    targets[possibleResults.IndexOf(input.Last())] = 0.99;

                    var inputList = input.Take(4).Select(double.Parse).ToArray();
                    network.Train(NormalizeIrisData(inputList), targets);
                }
            }

            var scoreCard = new List<bool>();

            var testDataset = shuffledInputs.Skip(100).ToArray();

            foreach (var input in testDataset)
            {
                var result = network.Query(NormalizeIrisData(input.Take(4).Select(double.Parse).ToArray())).ToList();
                var answer = possibleResults[possibleResults.IndexOf(input.Last())];
                var predictedIris = possibleResults[result.IndexOf(result.Max())];

                scoreCard.Add(answer == predictedIris);
            }

            Console.WriteLine($"Performance is {(scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)) * 100} percent");
        }

        private static string[][] Shuffle(string[][] allInputs)
        {
            var random = new Random();

            return allInputs.OrderBy(x => random.NextDouble()).ToArray();
        }

        private static double[] NormalizeIrisData(double[] input)
        {
            var maxSepalLenth = 7.9;
            //var minSepalLength = 4.3;

            var maxSepalWidth = 4.4;
            //var minSepalWidth = 2.0;

            var maxPetalLenth = 6.9;
            //var minPetalLength = 1.0;

            var maxPetalWidth = 2.5;
            //var minPetalWidth = 0.1;

            var normalized = new double[]
            {
                (input[0]/maxSepalLenth) + 0.01,
                (input[1]/maxSepalWidth) + 0.01,
                (input[2]/maxPetalLenth) + 0.01,
                (input[3]/maxPetalWidth) + 0.01
            };

            return normalized;
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
