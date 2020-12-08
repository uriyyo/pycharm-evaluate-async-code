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
                if (expression?.isAsyncCode == true) {
                    // FIXME: why so terrible? this code must be refactored
                    super.evaluate(PLUGIN, true, true)

                    // TODO: Does Kotlin has smth like python repr?
                    val fixedExpression = expression
                            .replace("'''", "\\'\\'\\'")
                            .let { "'''$it'''" }

                    val code = "__async_result__ = __import__('sys').__async_eval__($fixedExpression, globals(), locals())"
                    super.evaluate(code, true, doTrunc)

                    return super.evaluate("__async_result__", false, doTrunc)
                }

                return super.evaluate(expression, execute, doTrunc)
            }
        }
    }

}