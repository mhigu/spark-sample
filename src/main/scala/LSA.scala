import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object LSA{
  def main(args: Arrayp[String]){
    val conf = new SparkConf().setAppName("LSA").setMaster("local")
    val sc = new SparkContext(conf)
     

  }

  def termDocWeight(
    termFrequencyInDoc: Int,
    totalTermsInDoc: Int,
    termFreqInCorpus: Int,
    totalDocs: Int): Double = {
    val tf = termFrequencyInDoc.toDouble/ totalTermsInDoc
    val docFreq = totalDocs.toDouble/ termFreqInCorpus
    val idf = math.log(docFreq)
    tf * idf
  }

}
