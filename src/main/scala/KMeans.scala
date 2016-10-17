import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.rdd._

object KMeans{
  def main(args: Array[String]){

    val conf = new SparkConf().
      setAppName("KMeans").
      setMaster("local")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("src/main/resorces/data/kddcup.data")

    rawData.map(_.split(',').last).countByValue.toSeq.sortBy(_._2).reverse.foreach(println)
    val labelsAndData = rawData.map { line =>
      val buffer = line.split(",").toBuffer
      buffer.remove(1, 3)
      val label = buffer.remove(buffer.length-1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }

    val data = labelsAndData.values.cache()

    val kmeans = new KMeans()
    val model = kmeans.run(data)

    model.clusterCenters.foreach(println)

    val clusterLabelCount = labelsAndData.map { case(label, datum) =>
      val cluster = model.predict(datum)
      (cluster, label)
    }.countByValue

    clusterLabelCount.toSeq.sorted.foreach{
      case((cluster, label), count) => println(f"$cluster%1s$label%18s$count%8s")
    }

    (30 to 100 by 10).map(k => (k, clusteringScore(data, k))).foreach(println)


    val dataAsArray = data.map(_.toArray)
    val numCols = dataAsArray.first.length
    val n = dataAsArray.count
    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 + t._2)
    )
    val sumSquares = dataAsArray.fold(
      new Array[Double](numCols)
    )(
      (a,b) => a.zip(b).map(t => t._1 + t._2 * t._2)
    )
    val stdevs = sumSquares.zip(sums).map{
      case (sumSq, sum) => math.sqrt(n*sumSq - sum*sum)/n
    }
    val means = sums.map(_ / n)

    def normalize(datum: Vector) = {
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) => if (stdev<=0) (value -mean) else (value - mean) / stdev
      )
      Vectors.dense(normalizedArray)
    }
  

    val normalizedData = data.map(normalize).cache()

    (60 to 120 by 10).par.map(k =>
      (k, clusteringScore(normalizedData, k))).toList.foreach(println)

  }


  def distance(a: Vector, b: Vector) = math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d=> d * d).sum)

  def distToCentriod(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  def clusteringScore(data: RDD[Vector], k:Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setRuns(10)
    kmeans.setEpsilon(1.0e-4)
    val model = kmeans.run(data)
    data.map(datum => distToCentriod(datum, model)).mean()
  }


}
