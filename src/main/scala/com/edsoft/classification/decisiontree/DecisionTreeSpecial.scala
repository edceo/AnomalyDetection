package com.edsoft.classification.decisiontree

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by kora on 27.05.2016.
  */
class DecisionTreeSpecial extends Serializable {
  def dataFormat(rawData: RDD[String]) = {
    val protocols = rawData.map(
      _.split(",")(1)).distinct().collect().zipWithIndex.toMap
    rawData.map { line =>
      val buffer = ArrayBuffer[String]()
      buffer.appendAll(line.split(","))
      val protocol = buffer.remove(1)
      buffer.remove(1, 3)
      //buffer.filter(f => (f, buffer.length-1)=="normal" )
      val label = buffer.remove(buffer.length - 1)
      val newLabel = labelToDouble(label)
      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      //buffer.insertAll(1, newProtocolFeatures.map(_.toString))
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      LabeledPoint(newLabel, vector)
    }
  }

  private def labelToDouble(label: String): Double = {
    if (label.equals("normal.")) {
      return 1.0
    }
    0.0
  }

  def   anomalyDetectionWithDecisionTree(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): Unit = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 100

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    val prediction = model.predict(testData.map(p => p.features))
    val labelAndPrediction = testData.map(f => f.label).zip(prediction)

    val testAccuracy = labelAndPrediction.filter {
      case (l, p) => l == p
    }.count() / testData.count().toFloat

    println(testAccuracy)
    println(model.toDebugString)
  }
}
