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

    if (args.length != 4) {
      System.err.println(
        "Usage: MultivariateGaussianDistribution " +
          "<dataFile> <cvDataFile> <cvPredictionsFile> <outputDir>")
      System.exit(1)
    }

    val dataFile = args(0) //original data
    val cvDataFile = args(1) //tested data
    val cvPredictions = args(2) //lable
    val outputDir = args(3) //where to save the output results

    val conf = new SparkConf().setMaster("local[10]").setAppName("KDD Cup Example Multivariate Gaussian")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("/home/edsoft/KDD/SparkExample/src/main/resources/kddcup.data", 120)
    val multiSpecial = new MultivariateSpecial()

    val data = multiSpecial.dataFormat(rawData)

    val dataArray = data.map(_._2.toArray)

    val m = dataArray.count()
    val n = dataArray.first().length


    val sums = dataArray.reduce((a, b) => a.zip(b).map(t => t._1 + t._2))
    val mean = sums.map(_ / m)


    val subMuSquares = dataArray.aggregate(new Array[Double](n))((a, b) => a.zip(b).zipWithIndex.map { t =>
      val diffMu = t._1._2 - mean(t._2)
      t._1._1 + (diffMu * diffMu)
    }, (acc1, acc2) => acc1.zip(acc2).map(a => a._1 + a._2))
    val sigma2 = subMuSquares.map(_ / m)

    val multivariateGaussian = new MultivariateGaussian(Vectors.dense(mean), Matrices.diag(Vectors.dense(sigma2)))
    val ps = data.values.map(multivariateGaussian.pdf)


    val inputCV_X = sc.textFile(cvDataFile)
    val inputCV_Y = sc.textFile(cvPredictions)
    val examplesCV_X = inputCV_X.filter(s => s.length > 0 && !s.startsWith("#"))
    val examplesCV_Y = inputCV_Y.filter(s => s.length > 0 && !s.startsWith("#"))

    // Examples for cross-validation set along with "ground truth" for each example, i.e. explicitly marked as anomalous/non-anomalous
    val examplesCV = examplesCV_X.zip(examplesCV_Y).map {
      case (l1, l2) => LabeledPoint(l2.trim.toDouble, Vectors.dense(l1.trim.split(' ').map(_.toDouble)))
    }.persist()

    // Vector of probability density for cross validation set using learned parameters
    val psLabCV = examplesCV.map(lp => (multivariateGaussian.pdf(lp.features), lp.label))
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
