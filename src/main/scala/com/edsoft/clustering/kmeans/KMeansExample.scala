package com.edsoft.clustering.kmeans

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by edsoft on 3/21/16.
  */
object KMeansExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[4]").setAppName("KMeansExample")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/home/edsoft/KDD/SparkExample/src/main/resources/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //clusters.save(sc, "myModelPath")



    clusters.clusterCenters.foreach(f => println(f))

    //val sameModel = KMeansModel.load(sc, "myModelPath")
  }
}
