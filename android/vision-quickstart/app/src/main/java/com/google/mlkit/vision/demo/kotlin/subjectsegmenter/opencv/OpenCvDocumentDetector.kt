package com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv

import android.util.Log
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class OpenCvDocumentDetector {
    companion object {
        private const val TAG = "SubjectSegmentationGfx"
    }

    // Temporary data structures to be re-used for performance purposes
    private lateinit var maskedFrame: Mat
    private lateinit var hierarchy: Mat

    private lateinit var contour: MatOfPoint
    private lateinit var kernel: Mat

    private val contours = mutableListOf<MatOfPoint>()

    private var currentContour: MatOfPoint? = null
    private val blurSize = Size(5.0, 5.0)

    private var isProcessing = false

    fun initialize() {
        maskedFrame = Mat()
        hierarchy = Mat()
        contour = MatOfPoint()
        kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            blurSize
        )
    }

    fun release() {
        maskedFrame.release()
        hierarchy.release()
        kernel.release()

        contour.release()
        currentContour = null

        contours.release()
    }

    operator fun invoke(
        foregroundMask: Array<IntArray>,
    ): Array<IntArray>? {
        if (isProcessing) {
            Log.d(TAG, "dropped frame")
            return currentContour?.toArrayList()
        }

        isProcessing = true

        // Resize frame
//        Imgproc.resize(
//            frame,
//            resizedFrame,
//            zeroSize,
//            0.5,
//            0.5
//        )
        maskedFrame = foregroundMask.toMat() // TODO release

        Imgproc.Canny(
            maskedFrame,
            maskedFrame,
            75.0,
            200.0
        )

        // Dilate the image to get a thin outline of the document (??)
//        Imgproc.dilate(
//            maskedFrame,
//            maskedFrame,
//            kernel
//        )

        // Find edge contours

//        var maxVal: Int = 0
//        for (i in 0 until maskedFrame.rows()) {
//            for (j in 0 until maskedFrame.cols()) {
//                val result = byteArrayOf(0)
//                maskedFrame.get(i, j, result)
//                maxVal = max(maxVal, result[0].toInt())
//            }
//        }
//
//        Log.d(TAG, "maxVal $maxVal")

        Imgproc.findContours(
            maskedFrame,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        contours.sortByDescending { Imgproc.contourArea(it) }
        val frameArea = maskedFrame.rows() * maskedFrame.cols()

        // Find the largest contour with 4 corners
        contours.firstNotNullOfOrNull { contour ->
            val contourFloat = MatOfPoint2f()
            contour.toMatOfPoint2F(contourFloat)

            // Approximate the contour
            val peri = Imgproc.arcLength(
                contourFloat,
                true
            )

            Imgproc.approxPolyDP(
                contourFloat,
                contourFloat,
                0.02 * peri,
                true
            )

            Log.d(TAG, "found contour ${contourFloat.rows()} points, ${Imgproc.contourArea(contourFloat)} / $frameArea area")

            if (
                contourFloat.rows() == 4
//                && Imgproc.contourArea(contourFloat) >= frameArea * 0.25
            ) {
                contourFloat
            } else {
                null
            }
        }
            ?.let { screenContour ->
                screenContour.toMatOfPoint(contour)

                currentContour = contour
            }
            ?: run {
                currentContour = null
            }

        contours.release()
        isProcessing = false
        return currentContour?.toArrayList()
    }
}
