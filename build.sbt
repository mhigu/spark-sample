name := "Simple ML Project"

version := "1.0"

scalaVersion := "2.10.6"

val sparkVersion = "2.0.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)

