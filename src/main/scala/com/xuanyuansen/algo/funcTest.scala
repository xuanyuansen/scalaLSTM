package com.xuanyuansen.algo
import breeze.linalg._
import breeze.numerics.exp

/**
  * Created by wangshuai on 16/7/14.
  */
object funcTest {
  def main(args: Array[String]) {
    val x = DenseVector.ones[Double](2)
    val y = DenseVector.ones[Double](2)
    println(x)
    println( 1.0 - x dot y)
    println( 1.0 - x :* y)
    println(DenseVector.vertcat(x,y))
    val Wf = DenseMatrix.rand[Double](4, 2)
    println(Wf)

    val input_dim = 4
    val out_dim = 2
    val concat_len = input_dim + out_dim

    val WW = DenseMatrix.ones[Double](out_dim, concat_len)//2*6
    val kk= DenseMatrix.ones[Double](concat_len, 1)
    println("WW * kk")
    val tmp= (WW * kk).asInstanceOf[DenseMatrix[Double]]

    println(tmp)

    val diffo = DenseMatrix.ones[Double](1,out_dim)
    val input_h = DenseMatrix.ones[Double](1,out_dim)
    val input_x = DenseMatrix.ones[Double](1,input_dim) + DenseMatrix.ones[Double](1,input_dim)

    // 1*2 1*2
    val dinputh =  diffo.t * input_h
    println(dinputh)//2*2

    //1*2 1*4
    val dinputx =  diffo.t * input_x
    println(dinputx)//2*4

    val out = DenseMatrix.horzcat(dinputx, dinputh)
    //concat 2*6
    println(out)

    val softmax = exp(kk)
    println(softmax)
    val sumsoft = sum(softmax)

    val finalout = softmax/sumsoft
    println( "finalout "+finalout.toString )
    val test = DenseMatrix.rand[Double](2,3)
    println(test)
    val idx_max = argmax(test)
    println(idx_max)
    println(max(test))
    val z = DenseMatrix.zeros[Double](2,3)
    z(idx_max) = 1
    println(z)
    println(sum(exp(test)) - max(test))

    println((0 until 4).reverse)

    val zout = 2.0 * DenseMatrix.ones[Double](2,3)
    println(zout)
    println(1.0 / zout + 6.0)

    val out1 = DenseMatrix.create(5, 1 , Array(1,2,3,4,5))
    println(out1)

  }
}
