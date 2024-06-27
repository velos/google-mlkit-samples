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
import androidx.annotation.RequiresApi
import androidx.core.graphics.withMatrix
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.subjectsegmenter.opencv.OpenCvDocumentDetector
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentationResult

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
    color = Color.BLUE
    strokeWidth = 10f
  }

  /** Draws the segmented background on the supplied canvas. */
  override fun draw(canvas: Canvas) {
    val bitmap =
      Bitmap.createBitmap(
        colorMask,
        imageWidth,
        imageHeight,
        Bitmap.Config.ARGB_8888
      )

    val matrix = Matrix(transformationMatrix)
    if (isRawSizeMaskEnabled) {
      matrix.preScale(scaleX, scaleY)
    }

    canvas.drawBitmap(bitmap, matrix, null)
    canvas.withMatrix(matrix) {
      drawPoints(contourPoints, paint)
    }

    bitmap.recycle()
  }

  init {
    this.imageWidth = imageWidth
    this.imageHeight = imageHeight

    isRawSizeMaskEnabled =
      imageWidth != overlay.imageWidth || imageHeight != overlay.imageHeight
    scaleX = overlay.imageWidth * 1f / imageWidth
    scaleY = overlay.imageHeight * 1f / imageHeight

    val edgeMask = edgeDetector.detect(
      imageWidth,
      imageHeight,
      imageWidth,
      segmentationResult.foregroundConfidenceMask!!
    )

    contourPoints = openCvDocumentDetector.detect(
        edgeMask,
        imageWidth,
        imageHeight
      )

    colorMask = edgeMask

    convertToColorArray(colorMask)
  }

  private fun convertToColorArray(mask: IntArray) {
    mask.forEachIndexed { index, i ->
      mask[index] = Color.argb(i, 255, 0, 0)
    }
  }

  companion object {
    private const val TAG = "SubjectSegmentationGfx"
  }
}
