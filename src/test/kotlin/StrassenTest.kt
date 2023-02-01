import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class StrassenTest {
    @Test
    fun testStrassenSimple() {
        val a = Matrix(
            listOf(
                listOf(1.0, 2.0), listOf(3.0, 4.0)
            )
        )
        val b = Matrix(
            listOf(
                listOf(5.0, 6.0), listOf(7.0, 8.0)
            )
        )
        val expected = Matrix(
            listOf(
                listOf(19.0, 22.0), listOf(43.0, 50.0)
            )
        )
        assertEquals(expected, strassen(a, b))
    }

    @Test
    fun testStrassenWithIncompatibleDimensions() {
        val a = Matrix(
            listOf(
                listOf(1.0, 2.0), listOf(3.0, 4.0)
            )
        )
        val b = Matrix(
            listOf(
                listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)
            )
        )

        assertThrows(IllegalArgumentException::class.java) { strassen(a, b) }
    }

    @Test
    fun testStrassenWithBigMatrices() {
        val n = 512
        val a = generateRandomMatrix(n, n)
        val b = generateRandomMatrix(n, n)

        var startTime = System.currentTimeMillis()
        strassen(a, b)
        var endTime = System.currentTimeMillis()
        println("Time taken by strassen: ${endTime - startTime}ms")

        startTime = System.currentTimeMillis()
        a * b
        endTime = System.currentTimeMillis()
        println("Time taken by traditional multiplication: ${endTime - startTime}ms")
        println("This test shows that strassen have pretty unpleasant constant, " +
                "although if we'll take the matrices of shape (1024, 1024) it will work faster than simple multiplication")
    }


    @Test
    fun testStrassenWithNotSquareMatrices() {
        val a = Matrix(
            listOf(
                listOf(1.0, 2.0), listOf(3.0, 4.0)
            )
        )
        val b = Matrix(
            listOf(
                listOf(5.0, 7.0, 9.0), listOf(8.0, 9.0, 10.0)
            )
        )
        val expected = a * b
        assertEquals(expected, strassen(a, b))
    }
}
