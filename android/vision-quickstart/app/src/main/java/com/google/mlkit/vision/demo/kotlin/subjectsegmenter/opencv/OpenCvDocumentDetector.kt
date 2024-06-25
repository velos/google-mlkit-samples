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
        private const val TAG = "OpenCvDocumentDetector"
    }

    // Temporary data structures to be re-used for performance purposes
    private lateinit var scaleMat: Mat
    private lateinit var resizedFrame: Mat
    private lateinit var maskedFrame: Mat
    private lateinit var kernel: Mat
    private lateinit var mask: Mat
    private lateinit var bgdModel: Mat
    private lateinit var fgdModel: Mat
    private lateinit var hierarchy: Mat

    private lateinit var warped: Mat
    private lateinit var perspectiveTransformDst: Mat

    private lateinit var contour: MatOfPoint

    private val contours = mutableListOf<MatOfPoint>()

    private val centerPoint: Point = Point(-1.0, -1.0)
    private val blurSize = Size(5.0, 5.0)
    private val contourColor = Scalar(0.0, 255.0, 0.0)
    private val zeroSize = Size()

    private var currentContour: MatOfPoint? = null

    private var isProcessing = false

    fun initialize() {
        scaleMat = Mat(4, 1, CvType.CV_32FC2).apply {
            put(0, 0, floatArrayOf(2f, 2f))
            put(1, 0, floatArrayOf(2f, 2f))
            put(2, 0, floatArrayOf(2f, 2f))
            put(3, 0, floatArrayOf(2f, 2f))
        }

        resizedFrame = Mat()
        maskedFrame = Mat()
        mask = Mat()
        hierarchy = Mat()
        contour = MatOfPoint()

        warped = Mat()
        perspectiveTransformDst = Mat(4, 1, CvType.CV_32FC2)

//        kernel = Mat.ones(5, 5, CvType.CV_8UC1)
        kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            blurSize
        )

        bgdModel = Mat.zeros(1, 65, CvType.CV_64FC1)
        fgdModel = Mat.zeros(1, 65, CvType.CV_64FC1)
    }

    fun release() {
        scaleMat.release()
        resizedFrame.release()
        kernel.release()
        mask.release()
        bgdModel.release()
        fgdModel.release()
        maskedFrame.release()
        hierarchy.release()

        warped.release()
        perspectiveTransformDst.release()

        contour.release()
        currentContour = null

        contours.release()
    }

    operator fun invoke(
        frame: Mat,
        applyGrabCut: Boolean = false,
        applyPerspectiveTransform: Boolean = false
    ): Mat {
        if (isProcessing) {
            Log.d(TAG, "dropped frame")
            currentContour?.let {
                frame.drawContour(it)
            }

            return frame
        }

        isProcessing = true

        // https://learnopencv.com/automatic-document-scanner-using-opencv/
        // https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

        // Resize frame
        Imgproc.resize(
            frame,
            resizedFrame,
            zeroSize,
            0.5,
            0.5
        )

        // Remove alpha channel
        Imgproc.cvtColor(
            resizedFrame,
            maskedFrame,
            Imgproc.COLOR_RGBA2RGB
        )

        // GrabCut is pretty slow (~3s per operation) so it will be very laggy when used live.
        // Hold the camera still for several seconds to see the result.
        if (applyGrabCut) {
            // Remove document foreground using dilate / erode
            Imgproc.morphologyEx(
                maskedFrame,
                maskedFrame,
                Imgproc.MORPH_CLOSE,
                kernel,
                centerPoint,
                3
            )

            // Remove background using GrabCut
            // This treats a 20px wide border of the frame as the background, and analyzes that to
            // generate a mask that separates the foreground from background.
            val rect = Rect(
                20,
                20,
                resizedFrame.cols() - 40,
                resizedFrame.rows() - 40
            )

            Imgproc.grabCut(
                maskedFrame,
                mask,
                rect,
                bgdModel,
                fgdModel,
                5,
                Imgproc.GC_INIT_WITH_RECT
            )

            maskedFrame.applyGrabCutMask(mask)
        }

        // Use Canny to find edges

        // Convert to grayscale
        Imgproc.cvtColor(
            maskedFrame,
            maskedFrame,
            Imgproc.COLOR_RGBA2GRAY
        )

        // Blur the image to remove noise
        Imgproc.GaussianBlur(
            maskedFrame,
            maskedFrame,
            blurSize,
            0.0
        )

        Imgproc.Canny(
            maskedFrame,
            maskedFrame,
            75.0,
            200.0
        )

        // Dilate the image to get a thin outline of the document (??)
        Imgproc.dilate(
            maskedFrame,
            maskedFrame,
            kernel
        )

        // Find edge contours

        var maxVal: Int = 0
        for (i in 0 until maskedFrame.rows()) {
            for (j in 0 until maskedFrame.cols()) {
                val result = byteArrayOf(0)
                maskedFrame.get(i, j, result)
                maxVal = max(maxVal, result[0].toInt())
            }
        }

        Log.d(TAG, "maxVal $maxVal")

        Imgproc.findContours(
            maskedFrame,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        contours.sortByDescending { Imgproc.contourArea(it) }
        val frameArea = resizedFrame.rows() * resizedFrame.cols()

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

            if (
                contourFloat.rows() == 4 &&
                Imgproc.contourArea(contourFloat) >= frameArea * 0.25
            ) {
                contourFloat
            } else {
                null
            }
        }
            ?.let { screenContour ->
                // Scale the contour to match the original image dimensions
                val scaledScreenContour = screenContour
                    .mul(scaleMat)

                // Perspective transform
                if (applyPerspectiveTransform) {
                    // Apply the four point transform to obtain a top-down view of the original image
                    fourPointTransform(
                        frame,
                        scaledScreenContour,
                        warped
                    )

                    // onCameraFrame() wants the frame to match the original dimensions exactly,
                    // so force the warped frame dimensions (which will mess up the aspect ratio)
                    // for the POC
                    Imgproc.resize(warped, frame, frame.size())
                } else {
                    scaledScreenContour.toMatOfPoint(contour)

                    currentContour = contour
                    frame.drawContour(contour)
                }

                scaledScreenContour.release()
            }
            ?: run {
                currentContour = null
            }

        contours.release()
        isProcessing = false
        return frame
    }

    private fun Mat.drawContour(contour: MatOfPoint) {
        Imgproc.drawContours(
            this,
            mutableListOf(contour),
            -1,
            contourColor,
            2
        )
    }

    private fun Mat.applyGrabCutMask(mask: Mat) {
        val tmpByteArray = byteArrayOf(0)

        (0 until mask.rows()).forEach { row ->
            (0 until mask.cols()).forEach { col ->
                mask.get(row, col, tmpByteArray)
                val maskValue = tmpByteArray.first()

                if (maskValue == Imgproc.GC_BGD.toByte() || maskValue == Imgproc.GC_PR_BGD.toByte()) {
                    put(row, col, byteArrayOf(0, 0, 0))
                }
            }
        }
    }

    // Sorts the points by tl, tr, br, bl
    // https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py
    private fun Mat.orderPoints() {
        val x = listOf(
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
        )

        get(0, 0, x[0])
        get(1, 0, x[1])
        get(2, 0, x[2])
        get(3, 0, x[3])

        val xSorted = x.sortedBy { it[0] }

        val leftMost = xSorted.subList(0, 2)
        val rightMost = xSorted.subList(2, 4)

        val leftMostSorted = leftMost
            .sortedBy { it[1] }

        val tl = leftMostSorted[0]
        val bl = leftMostSorted[1]

        val rightMostSorted = rightMost
            .sortedBy {
                hypotenuse(
                    it[0] - tl[0],
                    it[1] - tl[1]
                )
            }

        val tr = rightMostSorted[0]
        val br = rightMostSorted[1]

        reshape(1, 4)
        put(0, 0, tl)
        put(1, 0, tr)
        put(2, 0, br)
        put(3, 0, bl)
    }

    // https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py
    private fun fourPointTransform(
        image: Mat,
        points: Mat,
        warped: Mat
    ) {
        points.orderPoints()

        val tl = points[0, 0]
        val tr = points[1, 0]
        val br = points[2, 0]
        val bl = points[3, 0]

        val widthA = hypotenuse(
            br[0] - bl[0],
            br[1] - bl[1]
        )
        val widthB = hypotenuse(
            tr[0] - tl[0],
            tr[1] - tl[1]
        )
        val maxWidth = max(widthA.toInt(), widthB.toInt())

        val heightA = hypotenuse(
            tr[0] - br[0],
            tr[1] - br[1]
        )
        val heightB = hypotenuse(
            tl[0] - bl[0],
            tl[1] - bl[1]
        )
        val maxHeight = max(heightA.toInt(), heightB.toInt())

        perspectiveTransformDst.put(0, 0, floatArrayOf(0f, 0f))
        perspectiveTransformDst.put(1, 0, floatArrayOf((maxWidth - 1).toFloat(), 0f))
        perspectiveTransformDst.put(2, 0, floatArrayOf((maxWidth - 1).toFloat(), (maxHeight - 1).toFloat()))
        perspectiveTransformDst.put(3, 0, floatArrayOf(0f, (maxHeight - 1).toFloat()))

        val m = Imgproc.getPerspectiveTransform(
            points,
            perspectiveTransformDst
        )

        Imgproc.warpPerspective(
            image,
            warped,
            m,
            Size(maxWidth.toDouble(), maxHeight.toDouble())
        )
    }
}
