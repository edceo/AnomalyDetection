package com.edsoft.clustering.kmeans.streaming_kmeans

import com.edsoft.clustering.kmeans.KMeansSpecial
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by kora on 27.05.2016.
  */
object KDDExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Streaming K-Means")
    val ssc = new StreamingContext(conf, Seconds(5))
    val kMeansSpecial = new KMeansSpecial()

    val trainingData = ssc.textFileStream("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/kddcupnormal.data")
      .transform(kMeansSpecial.dataFormat(_))
    val testData = ssc.textFileStream("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/kddanomalytest100.data")
      .transform(kMeansSpecial.dataFormat(_))
    val trData = trainingData.transform(f => f.values)
    trData.foreachRDD(f => println(f))
    val normalizedTrData = trData.transform(kMeansSpecial.dataNormalize(_))

    val tsData = testData.transform(f => f.values)
    val normalizedTsData = tsData.transform(kMeansSpecial.dataNormalizeForTestedData(_, kMeansSpecial.means, kMeansSpecial.stdevs))
    kMeansSpecial.anomalyDetectionWithStreamingKMeans(ssc, normalizedTrData, normalizedTsData)


  }
}
