package com.uriyyo.evaluate_async_code

import com.intellij.openapi.module.Module
import com.intellij.openapi.project.Project
import com.jetbrains.python.console.PydevConsoleRunnerFactory

class AsyncPyDebugConsoleRunnerFactory : PydevConsoleRunnerFactory() {
    private val frameworkAwareFactory: PydevConsoleRunnerFactory? = loadClass("com.jetbrains.FrameworkAwarePythonConsoleRunnerFactory")

    override fun createConsoleParameters(project: Project, contextModule: Module?): ConsoleParameters {
        val params = frameworkAwareFactory
            ?.getMethod<ConsoleParameters>("createConsoleParameters", Project::class.java, Module::class.java)
            ?.invoke(project, contextModule)
            ?: super.createConsoleParameters(project, contextModule)

        return ConsoleParameters(
            params.project,
            params.sdk,
            params.workingDir,
            params.envs,
            params.consoleType,
            params.settingsProvider,
            arrayOf(
                    setupAsyncPyDevScript(),
                    *(params.setupFragment ?: arrayOf()),
                    cleanupAsyncPyDevScript(),
            ),
        )
    }
}