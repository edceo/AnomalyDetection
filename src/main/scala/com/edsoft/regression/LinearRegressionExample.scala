package com.edsoft.regression

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel}

/**
  * Created by edsoft on 3/21/16.
  */
object LinearRegressionExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[4]").setAppName("RegressionExample")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/home/edsoft/IdeaProjects/SparkExample/src/main/resources/regression.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    // Building the model
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map { case (v, p) => math.pow(v - p, 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    valuesAndPreds.foreach(f => println(f._1 + "\t" + f._2))

    // Save and load model
    /* model.save(sc, "myModelPath")
     val sameModel = LinearRegressionModel.load(sc, "myModelPath")*/


  }

}
