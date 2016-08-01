name := "scalaLSTM"

version := "1.0"

scalaVersion := "2.10.6"

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" % "scala-logging-slf4j_2.10" % "2.1.2",
  "org.scalatest" %% "scalatest" % "3.0.0-M16-SNAP5",
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12"
)