package com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv

import android.util.Log
import kotlin.math.pow
import kotlin.math.sqrt
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f

fun Mat.toMatOfPoint(dst: MatOfPoint) {
    convertTo(dst, CvType.CV_32SC2)
}

fun Mat.toMatOfPoint2F(dst: MatOfPoint2f) {
    convertTo(dst, CvType.CV_32FC2)
}

fun <T : Mat> MutableCollection<T>.release() {
    iterator().let { iterator ->
        while (iterator.hasNext()) {
            iterator.next().release()
            iterator.remove()
        }
    }
}

fun hypotenuse(a: Double, b: Double): Double =
    sqrt(a.pow(2) + b.pow(2))

fun hypotenuse(a: Float, b: Float): Float =
    sqrt(a.pow(2) + b.pow(2))

fun Array<IntArray>.toMat(): Mat {
    val cols = size
    val rows = first().size
    val mat = Mat(cols, rows, CvType.CV_8UC1)

    (0 until cols).forEach { x ->
        val col = this[x]

        (0 until rows).forEach { y ->
            val byteArray = byteArrayOf(col[y].toByte())
            mat.put(y, x, byteArray)
        }
    }

    return mat
}

fun Mat.to2D(): Array<IntArray> {
    val result = arrayListOf<IntArray>()
    (0 until rows()).forEach { y ->
        result.add(IntArray(cols()))
    }

    val byteArray = byteArrayOf(0)

    (0 until cols()).forEach { y ->
        (0 until rows()).forEach { x ->
            get(y, x, byteArray)
            result[x][y] = byteArray[0].toInt() //if (byteArray[0].toInt() != 0) 255 else 0
        }
    }

    return result.toTypedArray()
}


fun MatOfPoint.to2D(): Array<IntArray> {
    return (0 until rows()).map { x ->
        intArrayOf(0, 0).apply { get(x, 0, this) }
    }.toTypedArray()
}

fun IntArray.to2D(width: Int, height: Int): Array<IntArray> {
    var index = 0
    val result = arrayListOf<IntArray>()
    (0 until width).forEach { x ->
        result.add(IntArray(height))
    }

    (0 until height).forEach { y ->
        (0 until width).forEach { x ->
            result[x][y] = this[index]
            index++
        }
    }

    return result.toTypedArray()
}

fun Mat.print(tag: String) {
    val byteArray = byteArrayOf(0)

    (0 until rows()).forEach { y ->
        val string = StringBuilder("$y: ")
        (0 until cols()).forEach { x ->
            get(x, y, byteArray)
            string.append(byteArray[0].toInt()).append(" ")
        }
        Log.d(tag, string.toString())
    }
}

fun MatOfPoint2f.print(tag: String) {
    val floatArray = floatArrayOf(0f, 0f)
    val string = StringBuilder()

    (0 until cols()).forEach { x ->
        (0 until rows()).forEach { y ->
            get(x, y, floatArray)
            string.append("(${floatArray[0]}, ${floatArray[1]}) ")
        }
    }
    Log.d(tag, string.toString())
}
