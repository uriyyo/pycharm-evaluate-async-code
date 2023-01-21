package com.uriyyo.evaluate_async_code

import com.intellij.execution.ExecutionResult
import com.intellij.execution.target.value.constant
import com.intellij.openapi.project.Project
import com.intellij.xdebugger.XDebugSession
import com.jetbrains.python.debugger.PyDebugProcess
import com.jetbrains.python.debugger.PyDebugRunner
import com.jetbrains.python.run.PythonCommandLineState
import com.jetbrains.python.run.PythonExecution
import com.jetbrains.python.run.PythonScriptExecution
import com.jetbrains.python.sdk.PythonSdkUtil
import java.net.ServerSocket

class AsyncPyDebugRunner : PyDebugRunner() {

    override fun configureDebugParameters(
            project: Project,
            pyState: PythonCommandLineState,
            debuggerScript: PythonExecution,
            debuggerScriptInServerMode: Boolean
    ) {
        pyState.sdk?.whenSupport {
            if (!PythonSdkUtil.isRemote(pyState.sdk)) {
                if (debuggerScript is PythonScriptExecution) {
                    debuggerScript.pythonScriptPath?.also { debuggerScript.parameters.add(0, it) }
                    debuggerScript.pythonScriptPath = constant(asyncPyDevScript().absolutePath)
                }
            }
        }

        super.configureDebugParameters(project, pyState, debuggerScript, debuggerScriptInServerMode)
    }

    override fun createDebugProcess(
            session: XDebugSession,
            serverSocket: ServerSocket,
            result: ExecutionResult,
            pyState: PythonCommandLineState
    ): PyDebugProcess = AsyncPyDebugProcess(
        session,
        serverSocket,
        result.executionConsole,
        result.processHandler,
        pyState.isMultiprocessDebug,
    )
}