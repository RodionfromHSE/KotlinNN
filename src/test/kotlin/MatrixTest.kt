import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class MatrixTest {
    @Test
    fun testMatrixMultiplication() {
        val a = Matrix(listOf(listOf(1.0, 2.0), listOf(3.0, 4.0)))
        val b = Matrix(listOf(listOf(2.0, 0.0), listOf(1.0, 2.0)))
        val expectedResult = Matrix(listOf(listOf(4.0, 4.0), listOf(10.0, 8.0)))
        assertEquals(expectedResult, a * b)
    }

    @Test
    fun testMatrixMultiplicationIncompatibleDimensions() {
        val a = Matrix(listOf(listOf(1.0, 2.0)))
        val b = Matrix(listOf(listOf(2.0, 0.0)))
        assertThrows(IllegalArgumentException::class.java) { a * b }
    }

    @Test
    fun testMatrixAddition() {
        val a = Matrix(listOf(listOf(1.0, 2.0), listOf(3.0, 4.0)))
        val b = Matrix(listOf(listOf(2.0, 0.0), listOf(1.0, 2.0)))
        val expectedResult = Matrix(listOf(listOf(3.0, 2.0), listOf(4.0, 6.0)))
        assertEquals(expectedResult, a + b)
    }

    @Test
    fun testMatrixAdditionIncompatibleDimensions() {
        val a = Matrix(listOf(listOf(1.0, 2.0)))
        val b = Matrix(listOf(listOf(2.0, 0.0), listOf(1.0, 2.0)))
        assertThrows(IllegalArgumentException::class.java) { a + b }
    }

    @Test
    fun testMatrixSigmoid() {
        val a = Matrix(listOf(listOf(-1.0, 0.0, 1.0)))
        val expectedResult = Matrix(listOf(listOf(sigmoid(-1.0), sigmoid(0.0), sigmoid(1.0))))
        assertEquals(expectedResult, a.sigmoid())
    }

    @Test
    fun testMatrixReLU() {
        val a = Matrix(listOf(listOf(-1.0, 0.0, 1.0)))
        val expectedResult = Matrix(listOf(listOf(relu(-1.0), relu(0.0), relu(1.0))))
        assertEquals(expectedResult, a.relu())
    }
}
