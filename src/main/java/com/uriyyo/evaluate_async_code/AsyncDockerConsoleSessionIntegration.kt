package com.uriyyo.evaluate_async_code

import com.intellij.execution.ExecutionResult
import com.intellij.execution.runners.ExecutionEnvironment
import com.intellij.openapi.projectRoots.Sdk
import com.intellij.xdebugger.XDebugProcess
import com.intellij.xdebugger.XDebugProcessStarter
import com.intellij.xdebugger.XDebugSession
import com.intellij.xdebugger.XDebuggerManager
import com.jetbrains.python.debugger.PyDebugProcess
import com.jetbrains.python.debugger.PyDebugRunner
import com.jetbrains.python.debugger.PyDebugSessionFactory
import com.jetbrains.python.run.PythonCommandLineState


abstract class AbstractAsyncDebugSessionFactory : PyDebugSessionFactory() {
    abstract fun getIntegration(): PyDebugSessionFactory?

    abstract fun createDebugProcess(
        session: XDebugSession,
        result: ExecutionResult,
        host: String,
        port: Int
    ): PyDebugProcess

    override fun appliesTo(sdk: Sdk): Boolean {
        return getIntegration()
            ?.getMethodByName<Boolean>("appliesTo")
            ?.invoke(sdk)
            ?: false
    }

    override fun createSession(state: PythonCommandLineState, environment: ExecutionEnvironment): XDebugSession {
        val integration = getIntegration()!!
        val sdk = state.sdk
        val remoteSdkAdditionalData = sdk?.sdkAdditionalData

        val serverHost: String = loadClass<Any>("com.intellij.docker.remote.run.runtime.DockerUtil")!!
            .getMethodByName<String>("getDockerHost")
            .invoke(remoteSdkAdditionalData)
        val serverPort: Int = integration.getMethodByName<Int>("generateDebuggerPort").invoke()

        val result: ExecutionResult = integration
            .getMethodByName<ExecutionResult>("executeDebuggingScript")
            .invoke(state, remoteSdkAdditionalData, environment, serverPort)

        val project = environment.project

        return XDebuggerManager.getInstance(project).startSession(environment, object : XDebugProcessStarter() {
            override fun start(session: XDebugSession): XDebugProcess {
                val pyDebugProcess: PyDebugProcess = createDebugProcess(
                    session,
                    result,
                    serverHost,
                    serverPort
                )
                PyDebugRunner.createConsoleCommunicationAndSetupActions(
                    project,
                    result,
                    pyDebugProcess,
                    session
                )

                return pyDebugProcess
            }
        })
    }
}

class AsyncDockerComposeDebugSessionFactory : AbstractAsyncDebugSessionFactory() {
    private val dockerIntegration: PyDebugSessionFactory? =
        loadClass("com.jetbrains.python.docker.compose.debugger.PyDockerComposeDebugSessionFactory")

    override fun getIntegration(): PyDebugSessionFactory? = dockerIntegration

    override fun createDebugProcess(
        session: XDebugSession,
        result: ExecutionResult,
        host: String,
        port: Int
    ): PyDebugProcess =
        object : AsyncPyDebugProcess(session, result.executionConsole, result.processHandler, host, port) {
            override fun detachDebuggedProcess() {
                this.processHandler.waitFor()
            }
        }
}

class AsyncDockerDebugSessionFactory : AbstractAsyncDebugSessionFactory() {
    private val dockerIntegration: PyDebugSessionFactory? =
        loadClass("com.jetbrains.python.docker.compose.debugger.PyDockerComposeDebugSessionFactory")

    override fun getIntegration(): PyDebugSessionFactory? = dockerIntegration

    override fun createDebugProcess(
        session: XDebugSession,
        result: ExecutionResult,
        host: String,
        port: Int
    ): PyDebugProcess = AsyncPyDebugProcess(session, result.executionConsole, result.processHandler, host, port)
}