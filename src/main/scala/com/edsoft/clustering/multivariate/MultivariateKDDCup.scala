package com.edsoft.clustering.multivariate

import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by edsoft on 4/1/16.
  */
object MultivariateKDDCup {
  def main(args: Array[String]) {

    val outputDir = "/home/kora/" //add where to save the output results

    val conf = new SparkConf().setMaster("local[*]").setAppName("KDD Cup Example Multivariate Gaussian")
    val sc = new SparkContext(conf)

    //Add training and test file Path
    val rawData = sc.textFile("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/" +
      "kddcupnormal.data")
    val newData = sc.textFile("/home/kora/IdeaProjects/AnomalyDetection/src/main/resources/" +
      "kddanomalytest100.data")

    val multiSpecial = new MultivariateSpecial()

    val trainingData = multiSpecial.dataFormat(rawData)
    val testData = multiSpecial.dataFormat(newData)

    val trainingDataArray = trainingData.map(_.features.toArray)

    val m = trainingDataArray.count()
    val n = trainingDataArray.first().length


    val sums = trainingDataArray.reduce((a, b) => a.zip(b).map(t => t._1 + t._2))
    val mean = sums.map(_ / m)


    val subMuSquares = trainingDataArray.aggregate(new Array[Double](n))((a, b) => a.zip(b).zipWithIndex.map { t =>
      val diffMu = t._1._2 - mean(t._2)
      t._1._1 + (diffMu * diffMu)
    }, (acc1, acc2) => acc1.zip(acc2).map(a => a._1 + a._2))
    val sigma2 = subMuSquares.map(_ / m)

    val multivariateGaussian = new MultivariateGaussian(Vectors.dense(mean), Matrices.diag(Vectors.dense(sigma2)))
    val ps = trainingData.map(_.features).map(multivariateGaussian.pdf)

    // Examples for cross-validation set along with "ground truth" for each example, i.e. explicitly marked as anomalous/non-anomalous
    testData.persist()

    // Vector of probability density for cross validation set using learned parameters
    val psLabCV = testData.map(lp => (multivariateGaussian.pdf(lp.features), lp.label))
    val minPsCV = psLabCV.map(_._1).min()
    val maxPsCV = psLabCV.map(_._1).max()

    val step = (maxPsCV - minPsCV) / 1000
    val epsF1 = (minPsCV to maxPsCV by step).map { epsilon =>
      val predictions = psLabCV.map(t => (t._1 < epsilon, t._2 != 0.0))

      // True positives
      val tp = predictions.filter(p => p._1 && p._2).count()
      // False positives
      val fp = predictions.filter(p => p._1 && !p._2).count()
      // False Negatives
      val fn = predictions.filter(p => !p._1 && p._2).count()

      // Precision
      val prec = tp.toDouble / (tp + fp)
      // Recall
      val rec = tp.toDouble / (tp + fn)

      // F1 Score
      val f1 = (2 * prec * rec) / (prec + rec)
      (epsilon, f1)
    }

    val bestEpsF1 = epsF1.foldLeft((0.0, 0.0)) { (acc, a) => if (acc._2 > a._2) acc else a }
    val epsilon = bestEpsF1._1

    val outliers = ps.zipWithIndex.filter(_._1 < epsilon)
    println(f"Best epsilon found using cross-validation: $epsilon%e")
    println(f"Best F1 on Cross Validation Set: ${bestEpsF1._2}%f")
    println(f"Outliers found: ${outliers.count()}%d")


    ps.saveAsTextFile(s"${outputDir}/ps") //probabilty densty value for each data vector
    //contains actual output of the algorithm : index & data vector probability density value for each detected anomaly
    //examples.zipWithIndex().map(_.swap).join(outliers.map(_.swap)).saveAsTextFile(s"${outputDir}/outliers")
    //contains pairs of epsilon and epsilon value corresponding F1 score
    sc.parallelize(epsF1).saveAsTextFile(s"${outputDir}/eps_f1")
    sc.stop()
  }
}
