fun generateRandomMatrix(rows: Int, cols: Int) = Matrix(List(rows) {
    MutableList(cols) {
        Math.random() * 10 - 5 // [-5,5] TODO: Normal random:)
    }
})


class DenseLayer(val weights: Matrix, val biases: Matrix) {
    fun forward(inputs: Matrix) = inputs * weights + biases
}

class Network(
    private val inputLayer: DenseLayer,
    private val hiddenLayer: DenseLayer,
    private val outputLayer: DenseLayer
) {
    fun forward(inputs: Matrix): Matrix {
        var currentOutput = inputLayer.forward(inputs)
        currentOutput = hiddenLayer.forward(currentOutput).relu()
        currentOutput = outputLayer.forward(currentOutput).sigmoid()
        return currentOutput
    }
}

// TODO: Your input is vector...
fun main() {
    val inputShape = listOf(4, 4)
    val weightsShape = listOf(4, 4)
    val biasShape = listOf(4, 4)
    val inputLayer =
        DenseLayer(
            weights = generateRandomMatrix(weightsShape.first(), weightsShape.last()),
            biases = generateRandomMatrix(biasShape.first(), biasShape.last())
        )
    val hiddenLayer =
        DenseLayer(
            weights = generateRandomMatrix(weightsShape.first(), weightsShape.last()),
            biases = generateRandomMatrix(biasShape.first(), biasShape.last())
        )
    val outputLayer =
        DenseLayer(
            weights = generateRandomMatrix(weightsShape.first(), weightsShape.last()),
            biases = generateRandomMatrix(biasShape.first(), biasShape.last())
        )
    val network = Network(inputLayer, hiddenLayer, outputLayer)
    val inputs = generateRandomMatrix(inputShape.first(), inputShape.last())
    val output = network.forward(inputs)
    println("Input")
    println(inputs)
    println("Output")
    println(output)
}