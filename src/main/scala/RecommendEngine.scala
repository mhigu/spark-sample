import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation._

object SimpleApp{
  def main(args: Array[String]){

    // initialization
    val conf = new SparkConf().
      setAppName("Simple Application").
      setMaster("localhost")
      //setMaster("spark://192.168.0.100:7077")
    val sc = new SparkContext(conf)
    val dataRoot = "src/main/resorces/data/profiledata_06-May-2005/"


    // User/Artist Data Load.
    val rawUserArtistData = sc.textFile(dataRoot + "user_artist_data.txt")
    //println("User ID Stats: " + rawUserArtistData.map(_.split(" ")(0).toDouble).stats())
    //println("Artist ID Stats: " + rawUserArtistData.map(_.split(" ")(1).toDouble).stats())
    //println("Play Count Stats: " + rawUserArtistData.map(_.split(" ")(2).toDouble).stats())

    // Artist Data Load.
    val rawArtistData = sc.textFile(dataRoot + "artist_data.txt")
    val artistByID = rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty){
        None
      } else {
        try{
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }

    // Artist Alias Load.
    val rawArtistAlias = sc.textFile(dataRoot + "artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap {line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty){
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()

    // Show Artist Name By ID.
    println(artistByID.lookup(6803336).head)
    println(artistByID.lookup(1000010).head)


    // Create Recommendation Model.
    val bArtistAlias = sc.broadcast(artistAlias)
    val trainData = rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(" ").map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count.toDouble)
    }.cache()

    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    println("Model Feature: " + model.userFeatures.mapValues(_.mkString(", ")).first())


    // Check Created Model.
    val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).
      filter{ case Array(user,_,_) => user.toInt == 2093760 }

    val existingProducts =
      rawArtistsForUser.map{case Array(_, artist, _) => artist.toInt}.collect.toSet

    println("User 2093760 Artists =========== ")
    artistByID.filter{ case (id, name) => existingProducts.contains(id)}.values.collect.foreach(println)

    val recommentations = model.recommendProducts(2093760, 5)
    recommentations.foreach(println)

    val recommendedProductIDs = recommentations.map(_.product).toSet

    println("Recommended Artists for User 2093760")
    artistByID.filter { case (id, name) =>
      recommendedProductIDs.contains(id)
    }.values.collect.foreach(println)

    


  }
}
