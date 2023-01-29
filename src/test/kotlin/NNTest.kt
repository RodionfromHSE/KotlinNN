import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NNTests {
    @Test
    fun testGenerateRandomMatrix() {
        val rows = 3
        val cols = 4
        val matrix = generateRandomMatrix(rows, cols)
        assertEquals(rows, matrix.m)
        assertEquals(cols, matrix.n)
    }

    @Test
    fun testDenseLayerForward() {
        val weights = Matrix(listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        ))
        val biases = Matrix(listOf(
            listOf(0.1, 0.2, 0.3)
        ))
        val inputs = Matrix(listOf(
            listOf(1.0, 2.0, 3.0)
        ))
        val expectedOutput = Matrix(listOf(
            listOf(30.1, 36.2, 42.3)
        ))
        val denseLayer = DenseLayer(weights, biases)
        val output = denseLayer.forward(inputs)
        assertEquals(expectedOutput, output)
    }

    @Test
    fun testNetworkForward() {
        val inputWeights = Matrix(listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        ))
        val inputBiases = Matrix(listOf(
            listOf(0.1, 0.2, 0.3)
        ))
        val hiddenWeights = Matrix(listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        ))
        val hiddenBiases = Matrix(listOf(
            listOf(0.1, 0.2, 0.3)
        ))
        val outputWeights = Matrix(listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        ))
        val outputBiases = Matrix(listOf(
            listOf(0.1, 0.2, 0.3)
        ))
        val inputs = Matrix(listOf(
            listOf(1.0, 2.0, 3.0)
        ))
        val expectedOutput = Matrix(listOf(
            listOf(1.0, 1.0, 1.0)
        ))
        val inputLayer = DenseLayer(inputWeights, inputBiases)
        val hiddenLayer = DenseLayer(hiddenWeights, hiddenBiases)
        val outputLayer = DenseLayer(outputWeights, outputBiases)
        val network = Network(inputLayer, hiddenLayer, outputLayer)
        val output = network.forward(inputs)
        assertEquals(expectedOutput, output)
    }
}
