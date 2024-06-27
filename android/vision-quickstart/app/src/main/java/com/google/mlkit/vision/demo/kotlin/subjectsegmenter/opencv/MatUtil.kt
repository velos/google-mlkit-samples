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

fun MatOfPoint.toFloatArray(): FloatArray {
    val result = FloatArray(rows() * 2)
    val intArray = intArrayOf(0, 0)
    var index = 0

    (0 until rows()).map { x ->
        get(x, 0, intArray)
        result[index] = intArray[0].toFloat()
        result[index + 1] = intArray[1].toFloat()
        index += 2
    }

    return result
}

fun IntArray.toMat(
    width: Int,
    height: Int
): Mat {
    val result = Mat(height, width, CvType.CV_8UC1)
    var index = 0

    (0 until height).forEach { y ->
        (0 until width).forEach { x ->
            result.put(y, x, byteArrayOf(this[index].toByte()))
            index++
        }
    }

    return result
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
