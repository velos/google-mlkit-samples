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
import android.os.Build
import android.util.Log
import androidx.annotation.ColorInt
import androidx.annotation.RequiresApi
import com.google.common.math.DoubleMath.roundToInt
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv.OpenCvDocumentDetector
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
//  private val contourPoints: Array<IntArray>

  private val imageWidth: Int
  private val imageHeight: Int
  private val isRawSizeMaskEnabled: Boolean
  private val scaleX: Float
  private val scaleY: Float

  /** Draws the segmented background on the supplied canvas. */
  override fun draw(canvas: Canvas) {
    val bitmap =
      Bitmap.createBitmap(
        colorMask,
//        contourPointsToBitmapMask(contourPoints),
        imageWidth,
        imageHeight,
        Bitmap.Config.ARGB_8888
      )
    if (isRawSizeMaskEnabled) {
      val matrix = Matrix(getTransformationMatrix())
      matrix.preScale(scaleX, scaleY)
      canvas.drawBitmap(bitmap, matrix, null)
    } else {
      canvas.drawBitmap(bitmap, getTransformationMatrix(), null)
    }
    bitmap.recycle()
  }

  private fun detectContours(subjectMask: Array<IntArray>): Array<IntArray> {
    val contourPoints = openCvDocumentDetector(subjectMask) ?: emptyArray()
    Log.d(TAG, "contourPoints: ${contourPoints.joinToString { "(${it[0]}, ${it[1]})" }}")

    return contourPoints
  }

  private fun getSubjectMask(confidenceMask: FloatBuffer): Array<IntArray> {
    val foregroundMask = (0 until imageWidth)
      .map { IntArray(imageHeight) }
      .toTypedArray()

    for (y in 0 until imageHeight) {
      for (x in 0 until imageWidth) {
        val confidence = confidenceMask.get()
        foregroundMask[x][y] = if (confidence > 0.3) 255 else 0//(confidence * 255).roundToInt()
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
      val index = point[0] * imageHeight + point[1]
      array[index] = Color.argb(128, 255, 0, 0)
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

//    val subjectMask = getSubjectMask(segmentationResult.foregroundConfidenceMask!!)
//    contourPoints = detectContours(subjectMask)
    colorMask = edgeDetector.detect(
      imageWidth,
      imageHeight,
      imageWidth,
      segmentationResult.foregroundConfidenceMask!!
    )//maskColorsFromFloatBuffer(subjectMask)
  }

  companion object {
    private const val TAG = "SubjectSegmentationGfx"
  }
}
