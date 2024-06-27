package com.google.mlkit.vision.demo.kotlin.subjectsegmenter

import android.graphics.Color
import java.nio.FloatBuffer

fun FloatBuffer.toIntArray(width: Int, height: Int): IntArray {
    val intArray = IntArray(width * height)

    intArray.indices.forEach { index ->
        val value = get()
        intArray[index] = if (value > 0.3) 255 else 0
    }

    rewind()

    return intArray
}

fun Array<IntArray>.toIntArray(width: Int, height: Int): IntArray {
    val array = IntArray(width * height)
    var index = 0
    for (y in 0 until height) {
        for (x in 0 until width) {
            val col = this[x]
            array[index] = col[y]
            index++
        }
    }

    return array
}

fun IntArray.toColor() {
    forEachIndexed { index, i ->
        this[index] = Color.argb(i, 255, 0, 0)
    }
}
