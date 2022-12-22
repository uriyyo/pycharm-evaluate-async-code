package com.uriyyo.evaluate_async_code

import com.intellij.openapi.module.Module
import com.intellij.openapi.project.Project
import com.jetbrains.python.console.PyConsoleOptions
import com.jetbrains.python.console.PydevConsoleRunnerFactory

class AsyncPyDebugConsoleRunnerFactory : PydevConsoleRunnerFactory() {
    private val frameworkAwareFactory: PydevConsoleRunnerFactory? =
        loadClass("com.jetbrains.FrameworkAwarePythonConsoleRunnerFactory")

    override fun createConsoleParameters(project: Project, contextModule: Module?): ConsoleParameters {
        val settingsProvider = PyConsoleOptions.getInstance(project).pythonConsoleSettings
        val oldCustomStartScript = settingsProvider.customStartScript

        try {
            settingsProvider.customStartScript =
                arrayOf(setupAsyncPyDevScript(), oldCustomStartScript).filter { it.isNotEmpty() }.joinToString("\n")

            return frameworkAwareFactory
                ?.getMethodByName<ConsoleParameters>("createConsoleParameters")
                ?.invoke(project, contextModule)
                ?: super.createConsoleParameters(project, contextModule)
        } finally {
            settingsProvider.customStartScript = oldCustomStartScript
        }
    }
}