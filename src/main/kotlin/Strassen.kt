import java.lang.Integer.max
import kotlin.math.*

fun strassen(a: Matrix, b: Matrix): Matrix {
    if (a.n != b.m) {
        throw IllegalArgumentException("Incompatible matrix dimensions for multiplication.")
    }

    val n = max(a.m, max(a.n, b.n))
    val l = 2.0.pow(ceil(log2(n.toDouble())).toInt()).toInt()

    val aPad = expandMatrix(a, l, l)
    val bPad = expandMatrix(b, l, l)
    require(aPad.n == l)

    val result = strassenRecursive(aPad, bPad)

    return Matrix(result.data.subList(0, a.m).map { it.subList(0, b.n) })
}

fun strassenRecursive(a: Matrix, b: Matrix): Matrix {
    val n = a.n
    require((n and (n - 1)) == 0) { "Matrix dimension must be power of 2" }
    require((a.m == a.n) && (a.n == b.m) && (b.m == b.n)) { "Matrix must be square and compatible" }
    if (n == 1) {
        return Matrix(listOf(listOf(a.data[0][0] * b.data[0][0])))
    }

    val halfN = n / 2
    val a11 = Matrix(a.data.subList(0, halfN).map { it.subList(0, halfN) })
    val a12 = Matrix(a.data.subList(0, halfN).map { it.subList(halfN, n) })
    val a21 = Matrix(a.data.subList(halfN, n).map { it.subList(0, halfN) })
    val a22 = Matrix(a.data.subList(halfN, n).map { it.subList(halfN, n) })

    val b11 = Matrix(b.data.subList(0, halfN).map { it.subList(0, halfN) })
    val b12 = Matrix(b.data.subList(0, halfN).map { it.subList(halfN, n) })
    val b21 = Matrix(b.data.subList(halfN, n).map { it.subList(0, halfN) })
    val b22 = Matrix(b.data.subList(halfN, n).map { it.subList(halfN, n) })

    val p1 = strassenRecursive(a11 + a22, b11 + b22)
    val p2 = strassenRecursive(a21 + a22, b11)
    val p3 = strassenRecursive(a11, b12 - b22)
    val p4 = strassenRecursive(a22, b21 - b11)
    val p5 = strassenRecursive(a11 + a12, b22)
    val p6 = strassenRecursive(a21 - a11, b11 + b12)
    val p7 = strassenRecursive(a12 - a22, b21 + b22)

    val c11 = p1 + p4 - p5 + p7
    val c12 = p3 + p5
    val c21 = p2 + p4
    val c22 = p1 - p2 + p3 + p6

    return Matrix(List(n) { i ->
        List(n) { j ->
            when {
                i < halfN && j < halfN -> c11.data[i][j]
                i < halfN -> c12.data[i][j - halfN]
                j < halfN -> c21.data[i - halfN][j]
                else -> c22.data[i - halfN][j - halfN]
            }
        }
    })
}

fun expandMatrix(matrix: Matrix, m: Int, n: Int): Matrix {
    val expandedData = List(m) { i ->
        List(n) { j ->
            if (i < matrix.m && j < matrix.n) matrix.data[i][j] else 0.0
        }
    }
    return Matrix(expandedData)
}