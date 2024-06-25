package com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv

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
