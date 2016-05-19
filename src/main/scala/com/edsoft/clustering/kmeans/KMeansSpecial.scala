package com.edsoft.clustering.kmeans

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.math._
/**
  * Created by edsoft on 3/21/16.
  * Bir kümedeki vektörleri centroidden olan uzaklıkların toplamı = clustering score
  * Clustering score uygun küme sayısını(k değerini) bulmak için hesaplanır
  */
class KMeansSpecial() extends Serializable {

  var threshold: Double = 0.0
  var model: KMeansModel = null
  var means: Array[Double]=null
  var stdevs: Array[Double]=null


  //def getColumns(dataFrame: RDD[String])=dataFrame.map(r => r(0)).collect()
  /*def toRDD(m: Matrix): RDD[Vector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
    val vectors = rows.map(row => new DenseVector(row.toArray))
    sc.parallelize(vectors)
  }
  def getColumns(data: RDD[String])={
    val raw = data.map(_.toArray)
    val n = raw.first().length
    val rdds: Vector[RDD[String]] = (1 to n).map(i => raw
      .map(r => Try(r(i)))
      .filter(_.isSuccess)
      .map(_.get)
    ).toVector}*/

  private def distance(a: Array[Double], b: Array[Double]) =
    sqrt(a.zip(b).map(p => p._1 - p._2).map(d => d * d).sum)

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
      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      //buffer.insertAll(1, newProtocolFeatures.map(_.toString))
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
  }

  @deprecated("Old Version")
  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setEpsilon(1.0e-64)
    //data.foreach(println)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

  /**
    * Calculate model and threshold value
    *
    * @param normData Original Data
    * @param label    Original Data Label
    * @param k        Number of Cluster
    */
  def clusteringScore(normData: RDD[Vector], label: RDD[String], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    //kmeans.setRuns(10)
    kmeans.setEpsilon(1.0e-6)
    //data.foreach(println)
    model = kmeans.run(normData)
    val distances = normData.map(datum => distToCentroid(datum, model))
    threshold = distances.top(100).last
    println(threshold)


    /**
      * Calculate Entropy
      * We find the best number of clusters. Before runs algorithm
      * This code block runs with a for circle and calculate the best cluster number.
      * Example Code :
      * (10 to 150 by 10).map(k => (k, kMeansSpecial.clusteringScore(normalizedData, label, k)))
      * .foreach(println)
      */
    /* val labelsInCluster = normData.map(model.predict).zip(label).groupByKey().values
     val labelCounts = labelsInCluster.map(_.groupBy(l => l).map(t => t._2.size))
     val n = normData.count()
     labelCounts.map(m => m.sum * entropy(m)).sum / n*/
  }

  def anomalyDetection(newData: RDD[Vector],model: KMeansModel,threshold: Double) = {
    newData.filter(f => distToCentroid(f, model) > threshold)
  }

  private def distToCentroid(datum: Vector, model: KMeansModel) = {
    val centroid =
      model.clusterCenters(model.predict(datum))
    distance(centroid.toArray, datum.toArray)
  }

  /**
    * This function prepare data for calculating entropy
    *
    * @param datum  Original Vector
    * @param means  Mean of one vector
    * @param stdevs Standard Deviation of one vector
    * @return Normalize Vector
    */
  def normalize(datum: Vector, means: Array[Double], stdevs: Array[Double]) = {
    val norm = (datum.toArray, means, stdevs).zipped.map(
      (value, mean, stdev) =>
        if (stdev <= 0)
          value - mean
        else
          (value - mean) / stdev
    )
    Vectors.dense(norm)
  }

  /**
    * Normalize Data
    *
    * @param data Original Data
    * @return Normalized Data
    */
  def dataNormalize(data: RDD[Vector]) = {
    data.unpersist(true)
    val dataAsArray = data.map(_.toArray)
    val numCols = dataAsArray.first().length
    val n = dataAsArray.count()

    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 + t._2))

    val sumSquares =
      dataAsArray.fold(new Array[Double](numCols))(
        (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2)
      )

    stdevs = sumSquares.zip(sums).map {
      case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
    }
    means = sums.map(_ / n)
    data.map(this.normalize(_, means, stdevs))
  }
  def dataNormalizeForTestedData(data: RDD[Vector],means: Array[Double], stdevs: Array[Double]) = {
    data.unpersist(true)
    val dataAsArray = data.map(_.toArray)
    data.map(this.normalize(_, means, stdevs))
  }

  /**
    * Good clusters = low entropy clusters :))
    *
    * @param counts Label count in each group
    * @return Entropy value for each label
    */
  private def entropy(counts: Iterable[Int]) = {
    val values = counts.filter(_ > 0)
    val n: Double = values.sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  def KmeansWithAnomalyDetection(normalizedData: RDD[Vector],newData: RDD[Vector])={
    val kmeans = new KMeans()
    kmeans.setK(18)
    kmeans.setRuns(10)
    kmeans.setEpsilon(1.0e-6)
    val nModel=kmeans.run(normalizedData)
    //def distance(a: Array[Double],b: Array[Double])= sqrt(a.zip(b).map(p=>p._1-p._2).map(d=> d*d).sum)

    val centroids=nModel.clusterCenters
    val distances=normalizedData.map(datum => distToCentroid(datum, nModel))
    //val distances=normalizedData.map(datum => (distance(centroids(nModel.predict(datum),datum)),datum),datum)
    val n=distances.count()
    println("the distances count is")
    println(n)
    //val nThreshold = distances.top(39000).last
    val nThreshold =0.1


    //println("cluster centroids are:")
    //centroids.foreach(f => println(f))
    /* println("new data is:")
     newData.foreach(f => println(f))*/
    //val testMaodel=kmeans.run(newData)
    //val newDistances=newData.map(datum => distToCentroid(datum, testMaodel))

    val anomalies=newData.filter(d => distToCentroid(d,nModel) > nThreshold)
    val dis=newData.map(f=> distToCentroid(f,nModel))
    /*println("anomaly data is:")
    anomalies.foreach(f => println(f))*/
    println("the distnces are:")
    dis.foreach(f => println(f))
    println("the threshold value is:")
    println(nThreshold)
    println("the anomaly connections are:")
    val testedDataCount= newData.count()
    val anomaliesCount = anomalies.count()
    println(testedDataCount)
    println(anomaliesCount)
    //println(f"$testedDataCount%18s$anomaliesCount")
    anomalies.foreach(f => println(f))

  }

  /**
    * for counting the number of data vectors in each cluster
    * @param dataAndLabel
    * @param aModel
    */
  def clusterLableCount(dataAndLabel: RDD[(String,Vector)],aModel: KMeansModel)={
    val  clusterLabelCount = dataAndLabel.map{
      case (label,data) => (aModel.predict(data),label)
    }.countByValue
    clusterLabelCount.toList.sorted.foreach{
      case ((_,label),count) => println(f"$label%18s$count%8s")
    }
  }
}
