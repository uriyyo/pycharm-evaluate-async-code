package com.uriyyo.evaluate_async_code

import com.intellij.execution.process.ProcessHandler
import com.intellij.execution.ui.ExecutionConsole
import com.intellij.xdebugger.XDebugSession
import com.jetbrains.python.debugger.PyDebugProcess
import com.jetbrains.python.debugger.PyDebugValue
import java.net.ServerSocket


open class AsyncPyDebugProcess : PyDebugProcess {
    constructor(
        session: XDebugSession,
        serverSocket: ServerSocket,
        executionConsole: ExecutionConsole,
        processHandler: ProcessHandler?,
        multiProcess: Boolean
    ) : super(session, serverSocket, executionConsole, processHandler, multiProcess)

    constructor(
        session: XDebugSession,
        executionConsole: ExecutionConsole,
        processHandler: ProcessHandler?,
        serverHost: String, serverPort: Int
    ) : super(session, executionConsole, processHandler, serverHost, serverPort)

    override fun evaluate(expression: String?, execute: Boolean, doTrunc: Boolean): PyDebugValue {
        if (expression?.let { "async" in it || "await" in it } == true) {
            super.evaluate(pydevd_async_init(), true, false)
            return super.evaluate(expression, execute, doTrunc)
        }

        return super.evaluate(expression, execute, doTrunc)
    }
}