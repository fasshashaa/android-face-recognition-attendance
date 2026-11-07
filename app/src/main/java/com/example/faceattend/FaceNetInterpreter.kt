package com.example.faceattend

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class FaceNetInterpreter(context: Context) {

    private val interpreter: Interpreter
    private val inputSize: Int
    private val embeddingSize: Int

    init {
        // Jangan ditelan diam-diam: kalau model gagal dimuat, lebih baik langsung ketahuan.
        val model = FileUtil.loadMappedFile(context, "mobilefacenet.tflite")
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))

        // Baca ukuran input/output langsung dari model (bukan hardcode 112 / 192)
        val inShape = interpreter.getInputTensor(0).shape()   // [1, H, W, 3]
        val outShape = interpreter.getOutputTensor(0).shape() // [1, N]
        inputSize = inShape[1]
        embeddingSize = outShape[1]
    }

    // Mengubah bitmap wajah menjadi embedding (ukuran sesuai model, biasanya 192)
    // @Synchronized: Interpreter TFLite tidak thread-safe
    @Synchronized
    fun getFaceEmbedding(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val imgData = convertBitmapToByteBuffer(resized)
        val embedding = Array(1) { FloatArray(embeddingSize) }

        interpreter.run(imgData, embedding)
        return embedding[0]
    }

    fun close() {
        interpreter.close()
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                imgData.putFloat((((value shr 16) and 0xFF) - 127.5f) / 128.0f)
                imgData.putFloat((((value shr 8) and 0xFF) - 127.5f) / 128.0f)
                imgData.putFloat(((value and 0xFF) - 127.5f) / 128.0f)
            }
        }
        imgData.rewind()
        return imgData
    }

    companion object {
        // Jarak Euclidean (semakin kecil = semakin mirip)
        fun calculateDistance(emb1: FloatArray, emb2: FloatArray): Float {
            if (emb1.size != emb2.size) return Float.MAX_VALUE
            var sum = 0.0f
            for (i in emb1.indices) {
                val diff = emb1[i] - emb2[i]
                sum += diff * diff
            }
            return sqrt(sum.toDouble()).toFloat()
        }

        fun floatArrayToString(array: FloatArray): String = array.joinToString(",")

        fun stringToFloatArray(str: String): FloatArray {
            if (str.isBlank()) return FloatArray(0)
            return str.split(",").map { it.toFloat() }.toFloatArray()
        }
    }
}