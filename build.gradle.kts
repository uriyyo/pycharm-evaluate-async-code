plugins {
    java
    kotlin("jvm") version "1.7.22"
    id("org.jetbrains.intellij") version "1.11.0"
}


group = "com.uriyyo.evaluate_async_code"
version = "1.20"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.3.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.3.1")
}


intellij {
    version.set(project.properties["ideaVersion"].toString())
    pluginName.set("evaluate-async-code")
    downloadSources.set(project.properties["downloadIdeaSources"] == "true")
    updateSinceUntilBuild.set(false)
    plugins.add("terminal")
    if ("PC" in project.properties["ideaVersion"].toString()) {
        plugins.add("python-ce")
    } else if ("PY" in project.properties["ideaVersion"].toString()) {
        plugins.add("python")
    }
}

tasks {
    withType<JavaCompile> {
        sourceCompatibility = "11"
        targetCompatibility = "11"
    }
}