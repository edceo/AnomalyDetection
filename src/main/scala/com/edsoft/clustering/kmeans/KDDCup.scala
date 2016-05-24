package com.edsoft.clustering.kmeans

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by edsoft on 3/21/16.
  */
object KDDCup {
  def main(args: Array[String]) {
    val kMeansSpecial = new KMeansSpecial()
    val conf = new SparkConf().setMaster("local[10]").setAppName("KDD Cup Example")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("/home/khadija/Desktop/khadija/SparkExample/src/main/resources/kddcupnormal.data")
    //newDtata represents the tested data
    val newData = sc.textFile("/home/khadija/Desktop/khadija/SparkExample/src/main/resources/kddanomalytest100.data")
    //val rawData = sc.textFile("/home/edsoft/KDD/SparkExample/src/main/resources/normalData.data", 120)
    val dataAndLabel = kMeansSpecial.dataFormat(rawData)
    val data = dataAndLabel.values.cache()
    val label = dataAndLabel.keys.cache()
    val normalizedData = kMeansSpecial.dataNormalize(data)


    // kMeansSpecial.clusteringScore(normalizedData, label, 60)
    //kMeansSpecial.clusterLableCount(dataAndLabel,kMeansSpecial.model)
    //kMeansSpecial.clusteringScore(data, label, 100)
    //(5 to 55 by 10).map(k => (k, kMeansSpecial.clusteringScore(normalizedData, k))).foreach(println)

    /*
    * tested side
    *
    **/
    val newDataDataAndLabel = kMeansSpecial.dataFormat(newData)
    val newDataData = newDataDataAndLabel.values.cache()
    val newDataLabel = newDataDataAndLabel.keys.cache()
    //val newDatanormalized = kMeansSpecial.dataNormalize(newDataData)
    val newDatanormalized = kMeansSpecial.dataNormalizeForTestedData (newDataData,kMeansSpecial.means,kMeansSpecial.stdevs)

    //println("tested data is:")
    //newDatanormalized.foreach(f=>println(f))
    //println(newDatanormalized.count())
    //kMeansSpecial.model.clusterCenters.foreach(f => println(f))
    //println("tested data is:")
    //newDataData.foreach(f=>println(f))
    kMeansSpecial.KmeansWithAnomalyDetection(normalizedData,newDatanormalized)
    //val count=kMeansSpecial.anomalyDetection(newDataData,kMeansSpecial.model,kMeansSpecial.threshold).count()

    //println(count)
    //kMeansSpecial.anomalyDetection(NewDatanormalized).foreach(println)

  }
}
