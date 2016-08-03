package com.xuanyuansen.algo.application

import breeze.linalg.DenseMatrix
import com.xuanyuansen.algo.{simpleLossLayer, LstmNeuralNetwork}


/**
  * Created by wangshuai on 16/8/3.
  * lstm binary classification
  * or sequential classification?
  */
object BehaviorModel {

  /**
    * fromFile2Data
    *
    * @param filePath : path, file format: label\tword1,word2
    * @param dict : word_dic
    * @return
    */
  def fromFile2Data(filePath : String , dict : Map[String, DenseMatrix[Double]])
  : Seq[(Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]])] = {
    val out = scala.io.Source.fromFile(filePath).getLines().map{
      r=>
        val line = r.split("\t")
        val data = line
          .apply(1)
          .split(",")
          .map{
            k=> dict.getOrElse(k, dict.get("UNKNOWN").get)
          }.toSeq

        val label = if (line.head.toInt==1)
          new Array[Int](data.length).map{r=> DenseMatrix((1.0,0.0)).t}
        else
          new Array[Int](data.length).map{r=> DenseMatrix((0.0,1.0)).t}
        label.toSeq -> data
    }
    out.toSeq
  }

  def loadDict(filePath : String, wordDimension : Int) : Map[String, DenseMatrix[Double]] = {
    scala.io.Source.fromFile(filePath).getLines().map{
      r=>
        val line = r.split("\t")
        val vec = DenseMatrix.create(wordDimension, 1 , line
          .apply(1)
          .split(",")
          .slice(0, wordDimension)
          .map{k=>
            k.toDouble
          })
        line.head -> vec
    }.toMap

  }

  def main(args: Array[String]) {
    val simpleLSTM = new LstmNeuralNetwork(128, Seq(64,2), 2, new simpleLossLayer)
    val dict = this.loadDict("dict.data", wordDimension = 128)
    val data = this.fromFile2Data("train.file", dict)

    for(idx<- 0 to data.length){
      val currData = data.apply(idx)

      val loss = simpleLSTM.multilayer_backward_propagation(currData._2, currData._1)
      println(loss)
      simpleLSTM.LstmParams.foreach{
        k=>k.update_param_adadelta(0.95)
      }
    }

  }
}
