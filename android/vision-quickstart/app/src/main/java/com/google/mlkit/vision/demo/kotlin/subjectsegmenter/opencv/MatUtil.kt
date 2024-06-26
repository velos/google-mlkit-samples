package com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv

import android.util.Log
import kotlin.math.pow
import kotlin.math.sqrt
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f

fun Mat.toMatOfPoint(dst: MatOfPoint) {
    convertTo(dst, CvType.CV_32S)
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
    val mat = Mat(rows, cols, CvType.CV_8UC1)

    (0 until cols).forEach { x ->
        val col = this[x]

        (0 until rows).forEach { y ->
            val byteArray = byteArrayOf(col[y].toByte())
            mat.put(x, y, byteArray)
        }
    }

    return mat
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


fun MatOfPoint.toArrayList(): Array<IntArray> {
    return (0 until rows()).map { row ->
        intArrayOf(0, 0).apply { get(row, 0, this) }
    }.toTypedArray()
}
