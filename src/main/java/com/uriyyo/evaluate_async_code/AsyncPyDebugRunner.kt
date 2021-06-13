package com.uriyyo.evaluate_async_code

import com.intellij.execution.ExecutionResult
import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.execution.configurations.ParamsGroup
import com.intellij.openapi.project.Project
import com.intellij.xdebugger.XDebugSession
import com.jetbrains.python.debugger.PyDebugProcess
import com.jetbrains.python.debugger.PyDebugRunner
import com.jetbrains.python.debugger.PyDebugValue
import com.jetbrains.python.run.PythonCommandLineState
import com.jetbrains.python.sdk.PythonSdkUtil
import java.net.ServerSocket

class AsyncPyDebugRunner : PyDebugRunner() {

    override fun configureDebugParameters(
            project: Project,
            debugParams: ParamsGroup,
            pyState: PythonCommandLineState,
            cmd: GeneralCommandLine
    ) {
        pyState.sdk?.whenSupport {
            if (!PythonSdkUtil.isRemote(pyState.sdk)) {
                debugParams.addPyDevAsyncWork()
            }
        }

        super.configureDebugParameters(project, debugParams, pyState, cmd)
    }

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
                if (expression?.let { "async" in it || "await" in it } == true) {
                    super.evaluate(pydevd_async_init(), true, false);
                }

                return super.evaluate(expression, execute, doTrunc)
            }
        }
    }

}