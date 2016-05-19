package com.edsoft.histogram

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SQLContext

import scala.collection.mutable.ArrayBuffer

/**
  * Created by edsoft on 5/5/16.
  */
object HistogramExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Histogram Example").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val rawData = sc.textFile("/media/edsoft/34EEB3D1EEB3899E/Users/EDSOFT/Desktop/Student Redention Predictor/DataSet/Kutsal dosya_Tüm öğrenciler_(Allerletzte)_fresh_25 11 14.csv")

    case class Student(tcNo: String, deptName: String, GradYear: Int, Stand: String, gpa: Double)
    val studentData = rawData.filter(line => line.split(",")(12) != "").map { line =>
      val buffer = new ArrayBuffer[String]
      buffer.appendAll(line.split(","))
      val newBuffer = new ArrayBuffer[String]
      newBuffer.insert(0, buffer.apply(1))
      newBuffer.insert(1, buffer.apply(3))
      newBuffer.insert(2, buffer.apply(6))
      newBuffer.insert(3, buffer.apply(7))
      newBuffer.insert(4, buffer.apply(12))
      new Student(buffer.apply(1), buffer.apply(3), Integer.parseInt(buffer.apply(6)), buffer.apply(7), java.lang.Double.parseDouble(buffer.apply(12)))
    }
    val gpa = studentData.map(line => line.gpa)

    println(gpa.histogram(10))
  }
}
