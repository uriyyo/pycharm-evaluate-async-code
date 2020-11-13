package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.RunProfile
import com.intellij.openapi.project.Project
import com.jetbrains.python.debugger.PyDebugRunner
import com.jetbrains.python.run.CommandLinePatcher
import com.jetbrains.python.run.PythonCommandLineState

class AsyncPyDebugRunner : PyDebugRunner() {
    companion object {
        init {
            registerPythonConsoleRunnerFactory()
        }
    }

    override fun createCommandLinePatchers(
            project: Project?,
            state: PythonCommandLineState?,
            profile: RunProfile?,
            serverLocalPort: Int
    ): Array<CommandLinePatcher> {
        return arrayOf(
                *super.createCommandLinePatchers(project, state, profile, serverLocalPort),
                CommandLinePatcher {
                    if (isSupportedVersion(state?.sdk?.versionString))
                        it.patchCommandLine("Debugger", "debug")
                }
        )
    }
}