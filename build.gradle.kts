import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    java
    kotlin("jvm") version "1.3.61"
    id("org.jetbrains.intellij") version "0.4.16"
}


group = "com.uriyyo.evaluate_async_code"
version = "1.1"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.3.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.3.1")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "11"
}


intellij {
    version = project.properties["ideaVersion"].toString()
    pluginName = "evaluate-async-code"
    downloadSources = project.properties["downloadIdeaSources"] == "true"
    updateSinceUntilBuild = false
    setPlugins("terminal")
    if ("PC" in project.properties["ideaVersion"].toString()) {
        setPlugins("python-ce")
    } else if ("PY" in project.properties["ideaVersion"].toString()) {
        setPlugins("python")
    }
}