package com.edsoft.clustering.multivariate

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by edsoft on 4/1/16.
  */
class MultivariateSpecial extends Serializable {
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
    if (label.equals("normal")) {
      return 1.0
    }
    0.0
  }
}
