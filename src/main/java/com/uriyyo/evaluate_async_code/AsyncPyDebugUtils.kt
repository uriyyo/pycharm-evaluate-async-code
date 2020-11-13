package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.application.ApplicationManager
import com.jetbrains.python.console.PythonConsoleRunnerFactory
import com.jetbrains.python.psi.LanguageLevel
import org.picocontainer.MutablePicoContainer

fun GeneralCommandLine.patchCommandLine(group: String, type: String) {
    environment["_ASYNC_PY_DEBUG_RUN_TYPE"] = type

    this
            .parametersList
            .paramsGroups
            .firstOrNull { it.id == group }
            ?.parametersList
            ?.set(0, AsyncPyDebugEntryPoint.file)
}

fun registerPythonConsoleRunnerFactory() {
    // FIXME: I believe it's a worst way to register AsyncPyDebugConsoleRunnerFactory implementation
    //  there must be better way to do it but Google didn't help to find solution :(
    val container = ApplicationManager.getApplication().picoContainer as MutablePicoContainer

    container.unregisterComponent(PythonConsoleRunnerFactory::class.java)

    container.registerComponentImplementation(
            PythonConsoleRunnerFactory::class.java,
            AsyncPyDebugConsoleRunnerFactory::class.java
    )
}

fun isSupportedVersion(version: String?): Boolean =
        version !== null && LanguageLevel
                .fromPythonVersion(version.split(" ").last())
                .isAtLeast(LanguageLevel.PYTHON36)
