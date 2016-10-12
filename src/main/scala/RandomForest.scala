import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object RandomForest{
  def main(args: Array[String]){

    // Create Spark Context
    val conf = new SparkConf().
      setAppName("RandomForest").
      setMaster("local")
    val sc = new SparkContext(conf)

    // Create Input Data
    val rawData = sc.textFile("src/main/resorces/data/covtype.data")
    val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }

    // Separate Data
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache
    cvData.cache
    testData.cache

    import org.apache.spark.mllib.evaluation._
    import org.apache.spark.mllib.tree._
    import org.apache.spark.mllib.tree.model._
    import org.apache.spark.rdd._

    def getMetrics(model: DecisionTreeModel, data:RDD[LabeledPoint]): MulticlassMetrics = {
      val predictionsAndLabels = data.map(example =>
        (model.predict(example.features), example.label)
      )
      new MulticlassMetrics(predictionsAndLabels)
    }

    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)
    val metrics = getMetrics(model, cvData)

    println(metrics.confusionMatrix)
    println(metrics.precision)

    println("########################")
    (0 until 7).map(
      cat => (metrics.precision(cat), metrics.recall(cat))
    ).foreach(println)

    def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
      val countsByCategory = data.map(_.label).countByValue()
      val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
      counts.map(_.toDouble / counts.sum)
    }

    val trainPriorProbabilities = classProbabilities(trainData)
    val cvPriorProbabilities = classProbabilities(cvData)
    trainPriorProbabilities.zip(cvPriorProbabilities).map {
      case(trainProb, cvProb) => trainProb * cvProb
    }.sum

    //println(cvPriorProbabilities)

    val evaluations =
      for (impurity <- Array("gini", "entropy"); depth <- Array(1, 20); bins <- Array(10, 300))
      yield {
        val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, depth, bins)
        val predictionsAndLabels = cvData.map(example =>
          (model.predict(example.features), example.label)
        )
        val accuracy = new MulticlassMetrics(predictionsAndLabels).precision
        ((impurity, depth, bins), accuracy)
      }

    println("@@@@@@@@@@@@@@@@@@@@@@@")
    evaluations.sortBy(_._2).reverse.foreach(println)

    val model_2 = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)

  }

}
