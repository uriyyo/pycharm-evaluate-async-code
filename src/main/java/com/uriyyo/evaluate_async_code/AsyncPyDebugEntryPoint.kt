package com.uriyyo.evaluate_async_code

import java.io.InputStream

// FIXME: it's a workaround to start _async_pydevd.py script with python
//      before start it copy content of a file to temporary file
//      then run this file using python interpreter (it is the best solution that I found)
class AsyncPyDebugEntryPoint {
    companion object {
        private var tempFile: String? = null

        val file
            get(): String {
                if (tempFile === null) {
                    val stream: InputStream = AsyncPyDebugEntryPoint::class.java.getResourceAsStream("_async_pydevd.py")

                    val temp = createTempFile(suffix = ".py")
                    stream.copyTo(temp.outputStream())
                    tempFile = temp.absolutePath
                }

                return tempFile!! // Will never be null, at least I think so
            }
    }
}