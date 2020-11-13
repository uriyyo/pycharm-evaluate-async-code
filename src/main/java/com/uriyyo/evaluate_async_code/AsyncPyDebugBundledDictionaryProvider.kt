package com.uriyyo.evaluate_async_code

import com.intellij.spellchecker.BundledDictionaryProvider

class AsyncPyDebugBundledDictionaryProvider : BundledDictionaryProvider {
    override fun getBundledDictionaries() = arrayOf("_async_pydevd.py")
}