package com.uriyyo.evaluate_async_code

import com.intellij.execution.ExecutionResult
import com.intellij.xdebugger.XDebugSession
import com.jetbrains.python.debugger.PyDebugProcess
import com.jetbrains.python.debugger.PyDebugRunner
import com.jetbrains.python.debugger.PyDebugValue
import com.jetbrains.python.run.PythonCommandLineState
import java.net.ServerSocket

class AsyncPyDebugRunner : PyDebugRunner() {
    override fun createDebugProcess(
            session: XDebugSession,
            serverSocket: ServerSocket,
            result: ExecutionResult,
            pyState: PythonCommandLineState
    ): PyDebugProcess {
        if (!isSupportedVersion(pyState.sdk?.versionString))
            return super.createDebugProcess(session, serverSocket, result, pyState)

        return object : PyDebugProcess(
                session, serverSocket, result.executionConsole, result.processHandler, pyState.isMultiprocessDebug
        ) {
            override fun evaluate(expression: String?, execute: Boolean, doTrunc: Boolean): PyDebugValue {
                super.evaluate(PLUGIN, true, true)
                return super.evaluate(expression, execute, doTrunc)
            }
        }
    }

}