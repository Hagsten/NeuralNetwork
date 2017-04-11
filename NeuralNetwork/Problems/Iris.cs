using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Problems
{
    public static class Iris
    {
        private static readonly List<string> PossibleResults = new List<string> { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

        public static void Run()
        {
            var shuffledInputs = GetInputs();

            var network = new NeuralNetwork(4, 5, 3, 0.2);

            var trainDataSet = shuffledInputs.Take(100).ToArray();
            const int epochs = 500;

            Console.WriteLine($"Training network with {trainDataSet.Length} samples using {epochs} epochs...");

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var input in trainDataSet)
                {
                    var targets = new[] { 0.01, 0.01, 0.01 };
                    targets[PossibleResults.IndexOf(input.Last())] = 0.99;

                    var inputList = input.Take(4).Select(double.Parse).ToArray();
                    network.Train(NormalizeIrisData(inputList), targets);
                }
            }

            var scoreCard = new List<bool>();

            var testDataset = shuffledInputs.Skip(100).ToArray();

            foreach (var input in testDataset)
            {
                var result = network.Query(NormalizeIrisData(input.Take(4).Select(double.Parse).ToArray())).ToList();
                var answer = PossibleResults[PossibleResults.IndexOf(input.Last())];
                var predictedIris = PossibleResults[result.IndexOf(result.Max())];

                scoreCard.Add(answer == predictedIris);
            }

            Console.WriteLine(
                $"Performance is {(scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)) * 100} percent.");
        }

        private static string[][] GetInputs()
        {
            var dataset = File.ReadAllLines(@"C:/Temp/iris.csv");

            var allInputs = dataset.Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            var shuffledInputs = Shuffle(allInputs);
            return shuffledInputs;
        }

        private static string[][] Shuffle(string[][] allInputs)
        {
            var random = new Random();

            return allInputs.OrderBy(x => random.NextDouble()).ToArray();
        }

        private static double[] NormalizeIrisData(double[] input)
        {
            var maxSepalLenth = 7.9;
            var maxSepalWidth = 4.4;
            var maxPetalLenth = 6.9;
            var maxPetalWidth = 2.5;

            var normalized = new[]
            {
                (input[0]/maxSepalLenth) + 0.01,
                (input[1]/maxSepalWidth) + 0.01,
                (input[2]/maxPetalLenth) + 0.01,
                (input[3]/maxPetalWidth) + 0.01
            };

            return normalized;
        }
    }
}
