using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Problems
{
    public static class Sine
    {
        public static void Run()
        {
            var network = new NeuralNetwork(1, 15, 1, 0.01);

            var shuffledInputs = GenerateShuffeledInputs();

            var trainData = shuffledInputs.Take(shuffledInputs.Length / 2).ToArray();
            var testData = shuffledInputs.Skip(shuffledInputs.Length / 2).ToArray();

            for (var epoch = 0; epoch < 100; epoch++)
            {
                foreach (var data in trainData)
                {
                    var normalizedInput = data / Math.PI; //normalized [-1, 1]
                    var actual = (Math.Sin(normalizedInput) + 1) / 2; //normalized between [0, 1]

                    network.Train(new[] { normalizedInput }, new[] { actual });
                }
            }

            foreach (var data in testData.OrderBy(x => x).ToArray())
            {
                var normalizedInput = data / Math.PI;
                var result = network.Query(new[] { normalizedInput });
                var errorMargin = ((result[0] * 2) - 1) - Math.Sin(normalizedInput);
                Console.WriteLine("Final error (target - actual): " + errorMargin);
            }
        }

        private static double[] GenerateShuffeledInputs()
        {
            var inputs = new List<double>();

            for (var x = -Math.PI; x < Math.PI; x += 0.005)
            {
                inputs.Add(x);
            }

            var random = new Random();

            return inputs.OrderBy(x => random.NextDouble()).ToArray();
        }
    }
}
