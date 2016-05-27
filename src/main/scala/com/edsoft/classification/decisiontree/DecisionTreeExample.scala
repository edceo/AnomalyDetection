package com.edsoft.classification.decisiontree

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by kora on 27.05.2016.
  */
object DecisionTreeExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("Decision Tree Anomaly Detection")
    val sc = new SparkContext(conf)

    val decisionTreeSpecial = new DecisionTreeSpecial

    val rawData = sc.textFile("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/"
      + "kddcupnormal.data")
    val newData = sc.textFile("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/"
      + "kddanomalytest100.data")

    val trainingData = decisionTreeSpecial.dataFormat(rawData)
    val testData = decisionTreeSpecial.dataFormat(newData)

    decisionTreeSpecial.anomalyDetectionWithDecisionTree(trainingData, testData)


  }
}
