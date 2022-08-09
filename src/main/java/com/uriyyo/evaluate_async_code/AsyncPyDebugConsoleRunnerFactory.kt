package com.uriyyo.evaluate_async_code

import com.intellij.openapi.module.Module
import com.intellij.openapi.project.Project
import com.jetbrains.python.console.PydevConsoleRunnerFactory

class AsyncPyDebugConsoleRunnerFactory : PydevConsoleRunnerFactory() {
    private val frameworkAwareFactory: PydevConsoleRunnerFactory? =
        loadClass("com.jetbrains.FrameworkAwarePythonConsoleRunnerFactory")

    override fun createConsoleParameters(project: Project, contextModule: Module?): ConsoleParameters {
        val params = frameworkAwareFactory
            ?.getMethodByName<ConsoleParameters>("createConsoleParameters")
            ?.invoke(project, contextModule)
            ?: super.createConsoleParameters(project, contextModule)

        return when (params) {
            is ConstantConsoleParameters -> ConstantConsoleParameters(
                params.project,
                params.sdk,
                params.workingDir,
                params.envs,
                params.consoleType,
                params.settingsProvider,
                arrayOf(setupAsyncPyDevScript(), *params.setupFragment),
            )
            else -> {
                ignoreExc {
                    val setupScript = params.getFieldVal<Any>("setupScript")
                    val oldValue = setupScript.getFieldVal<String>("s")
                    setupScript.setFieldValue("s", "${setupAsyncPyDevScript()}\n$oldValue")
                }

                params
            }
        }
    }
}