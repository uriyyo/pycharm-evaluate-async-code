package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.project.Project
import com.intellij.openapi.projectRoots.Sdk
import com.intellij.util.Consumer
import com.jetbrains.python.console.*

class AsyncPyDebugConsoleRunnerFactory : PydevConsoleRunnerFactory() {

    override fun createConsoleRunner(
            project: Project,
            sdk: Sdk?,
            workingDir: String?,
            envs: MutableMap<String, String>,
            consoleType: PyConsoleType, settingsProvider: PyConsoleOptions.PyConsoleSettings,
            rerunAction: Consumer<in String>,
            vararg setupFragment: String
    ): PydevConsoleRunner {
        return object : PydevConsoleRunnerImpl(
                project,
                sdk,
                consoleType,
                workingDir,
                envs,
                settingsProvider,
                rerunAction,
                *setupFragment
        ) {
            override fun doCreateConsoleCmdLine(
                    sdk: Sdk,
                    environmentVariables: MutableMap<String, String>,
                    workingDir: String?,
                    port: Int
            ): GeneralCommandLine =
                    super.doCreateConsoleCmdLine(sdk, environmentVariables, workingDir, port)
                            .apply {
                                sdk.whenSupport {
                                    parametersList
                                            .paramsGroups
                                            .firstOrNull { it.id == "Script" }
                                            ?.addPyDevAsyncWork()
                                }
                            }
        }
    }

}