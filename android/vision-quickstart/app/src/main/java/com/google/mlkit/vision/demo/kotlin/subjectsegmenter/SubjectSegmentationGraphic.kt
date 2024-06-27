/*
 * Copyright 2023 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.kotlin.subjectsegmenter

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Build
import android.util.Log
import androidx.annotation.ColorInt
import androidx.annotation.RequiresApi
import androidx.core.graphics.withMatrix
import com.google.common.math.DoubleMath.roundToInt
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv.OpenCvDocumentDetector
import com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv.to2D
import com.google.mlkit.vision.segmentation.subject.Subject
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentationResult
import java.nio.FloatBuffer
import kotlin.collections.List
import kotlin.math.min
import kotlin.math.roundToInt

/** Draw the mask from [SubjectSegmentationResult] in preview. */
@RequiresApi(Build.VERSION_CODES.N)
class SubjectSegmentationGraphic(
  private val openCvDocumentDetector: OpenCvDocumentDetector,
  private val edgeDetector: EdgeDetector,
  overlay: GraphicOverlay,
  segmentationResult: SubjectSegmentationResult,
  imageWidth: Int,
  imageHeight: Int
) : GraphicOverlay.Graphic(overlay) {
  private val colorMask: IntArray
  private val contourPoints: FloatArray

  private val imageWidth: Int
  private val imageHeight: Int
  private val isRawSizeMaskEnabled: Boolean
  private val scaleX: Float
  private val scaleY: Float
  private val paint = Paint().apply {
    color = Color.RED
    strokeWidth = 10f
  }

  /** Draws the segmented background on the supplied canvas. */
  override fun draw(canvas: Canvas) {
    val bitmap =
      Bitmap.createBitmap(
        colorMask,
//        IntArray(imageWidth * imageHeight),
        imageWidth,
        imageHeight,
        Bitmap.Config.ARGB_8888
      )

    val matrix = Matrix(getTransformationMatrix())
    if (isRawSizeMaskEnabled) {
      matrix.preScale(scaleX, scaleY)
//      canvas.drawBitmap(bitmap, matrix, null)
//    } else {
//      canvas.drawBitmap(bitmap, getTransformationMatrix(), null)
    }
    canvas.drawBitmap(bitmap, matrix, null)
    canvas.withMatrix(matrix) {
      drawPoints(contourPoints, paint)
//      drawLines(contourPoints, paint)
    }

    bitmap.recycle()
  }

  private fun detectContours(subjectMask: Array<IntArray>): FloatArray {
    val contourPoints = openCvDocumentDetector(subjectMask) ?: emptyArray()

    Log.d(TAG, "image: $imageWidth x $imageHeight mask: ${subjectMask.size} x ${subjectMask.first().size}")
    Log.d(TAG, "contourPoints: ${contourPoints.joinToString { "(${it[0]}, ${it[1]})" }}")

    val lines = FloatArray(contourPoints.size * 4)

    if (contourPoints.size > 1) {
      var index = 0
      contourPoints.forEachIndexed { i, ints ->
        lines[index] = ints[0].toFloat()
        index++
        lines[index] = ints[1].toFloat()
        index++

        if (i > 0) {
          lines[index] = ints[0].toFloat()
          index++
          lines[index] = ints[1].toFloat()
          index++
        }
      }
      lines[index] = contourPoints[0][0].toFloat()
      index++
      lines[index] = contourPoints[0][1].toFloat()
      index++

      Log.d(TAG, "lines: ${lines.joinToString()}")
    }
    return lines
  }

  private fun getSubjectMask(confidenceMask: FloatBuffer): Array<IntArray> {
    val foregroundMask = (0 until imageWidth)
      .map { IntArray(imageHeight) }
      .toTypedArray()

    for (y in 0 until imageHeight) {
      for (x in 0 until imageWidth) {
        val confidence = confidenceMask.get()
//        foregroundMask[x][y] = (confidence * 255).roundToInt()
        foregroundMask[x][y] = if (confidence > 0.3) 255 else 0
      }
    }

    confidenceMask.rewind()

    return foregroundMask
  }

  private fun convertForegroundMaskToColorMask(mask: Array<IntArray>): IntArray {
    val array = IntArray(imageWidth * imageHeight)
    var index = 0
    for (y in 0 until imageHeight) {
      for (x in 0 until imageWidth) {
        val col = mask[x]
        array[index] = col[y]
        index++
      }
    }

    return array
  }

  private fun contourPointsToBitmapMask(contourPoints: Array<IntArray>): IntArray {
    val array = IntArray(imageWidth * imageHeight)

    contourPoints.forEach { point ->
      val index = point[0] * imageWidth + point[1]
      array[index] = Color.argb(255, 255, 0, 0)
    }

    return array
  }

  /**
   * Converts [FloatBuffer] floats from all subjects to ColorInt array that can be used as a mask.
   */
  @ColorInt
  private fun maskColorsFromFloatBuffer(subjectMask: Array<IntArray>): IntArray {
    val colorArray = convertForegroundMaskToColorMask(subjectMask)

    colorArray.forEachIndexed { index, i ->
      colorArray[index] = Color.argb(i, 255, 255, 0)
    }

    return colorArray
  }

  init {
    this.imageWidth = imageWidth
    this.imageHeight = imageHeight

    isRawSizeMaskEnabled =
      imageWidth != overlay.imageWidth || imageHeight != overlay.imageHeight
    scaleX = overlay.imageWidth * 1f / imageWidth
    scaleY = overlay.imageHeight * 1f / imageHeight

    val subjectMask = getSubjectMask(segmentationResult.foregroundConfidenceMask!!)
    colorMask = maskColorsFromFloatBuffer(openCvDocumentDetector.debug(subjectMask))
    val edgeMask = edgeDetector.detect(
      imageWidth,
      imageHeight,
      imageWidth,
      segmentationResult.foregroundConfidenceMask!!
    ).to2D(imageWidth, imageHeight)
    contourPoints = detectContours(edgeMask)
//      colorMask = maskColorsFromFloatBuffer(openCvDocumentDetector.debug(subjectMask))
//    colorMask = maskColorsFromFloatBuffer(edgeMask)
  }

  companion object {
    private const val TAG = "SubjectSegmentationGfx"
  }
}
