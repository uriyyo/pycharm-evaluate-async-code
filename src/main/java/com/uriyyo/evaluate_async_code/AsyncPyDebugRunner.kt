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
        return object : PyDebugProcess(
                session, serverSocket, result.executionConsole, result.processHandler, pyState.isMultiprocessDebug
        ) {
            override fun evaluate(expression: String?, execute: Boolean, doTrunc: Boolean): PyDebugValue {
                if (expression?.isAsyncCode == true) {
                    super.evaluate(expression.toAsyncCode(), true, true)

                    val value = super.evaluate(ASYNC_RESULT_VAR, false, true)
                    super.evaluate(ASYNC_CLEAN_UP, true, true)

                    return value
                }

                return super.evaluate(expression, execute, doTrunc)
            }
        }
    }

}