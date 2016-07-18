package com.xuanyuansen.algo

import breeze.linalg._
import breeze.numerics.{exp, tanh, sigmoid}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by wangshuai on 16/7/14.
  * basic lstm network
  */
case class LSTMLayerParam(val input_dim: Int, val out_dim: Int){
  /**
    * for simple case, 1 step lstm, hidden_dim is output_dim
    */
  val  concat_len = input_dim + out_dim

  /**
    * f: forget gate
    * g: cell gate
    * o: output gate
    * i: input gate
    */
  var Wo = DenseMatrix.rand[Double](out_dim, concat_len)
  var Wf = DenseMatrix.rand[Double](out_dim, concat_len)
  var Wi = DenseMatrix.rand[Double](out_dim, concat_len)
  var Wg = DenseMatrix.rand[Double](out_dim, concat_len)

  var Bo = DenseMatrix.rand[Double](out_dim, 1)
  var Bf = DenseMatrix.rand[Double](out_dim, 1)
  var Bi = DenseMatrix.rand[Double](out_dim, 1)
  var Bg = DenseMatrix.rand[Double](out_dim, 1)

  /**
    *
    */
  var wo_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
  var wf_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
  var wi_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
  var wg_diff = DenseMatrix.zeros[Double](out_dim, concat_len)


  var bo_diff = DenseMatrix.zeros[Double](out_dim, 1)
  var bf_diff = DenseMatrix.zeros[Double](out_dim, 1)
  var bi_diff = DenseMatrix.zeros[Double](out_dim, 1)
  var bg_diff = DenseMatrix.zeros[Double](out_dim, 1)


  def update_param(lr : Double): Unit ={

    this.Wo -= this.wo_diff * lr
    this.Wf -= this.wf_diff * lr
    this.Wi -= this.wi_diff * lr
    this.Wg -= this.wg_diff * lr

    this.Bo -= this.bo_diff * lr
    this.Bf -= this.bf_diff * lr
    this.Bi -= this.bi_diff * lr
    this.Bg -= this.bg_diff * lr

    this.reset_diff()
  }

  def reset_diff(): Unit ={
    this.wo_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
    this.wf_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
    this.wi_diff = DenseMatrix.zeros[Double](out_dim, concat_len)
    this.wg_diff = DenseMatrix.zeros[Double](out_dim, concat_len)


    this.bo_diff = DenseMatrix.zeros[Double](out_dim, 1)
    this.bf_diff = DenseMatrix.zeros[Double](out_dim, 1)
    this.bi_diff = DenseMatrix.zeros[Double](out_dim, 1)
    this.bg_diff = DenseMatrix.zeros[Double](out_dim, 1)
  }
}


class LSTMLayerNode(val input_dim: Int, val out_dim: Int){
  /**
    * one state for each time,
    * we do not need to save these states in forward propagation,
    * but we need them in backward propagation.
    */
  var f = DenseMatrix.zeros[Double](out_dim, 1)
  var o = DenseMatrix.zeros[Double](out_dim, 1)
  var i = DenseMatrix.zeros[Double](out_dim, 1)
  var g = DenseMatrix.zeros[Double](out_dim, 1)

  var state_cell = DenseMatrix.zeros[Double](out_dim, 1)
  var state_h = DenseMatrix.zeros[Double](out_dim, 1)

  var diff_cell_t= DenseMatrix.zeros[Double](out_dim, 1)
  var bottom_diff_h_t_minus_1 = DenseMatrix.zeros[Double](out_dim, 1)
  var bottom_diff_cell_t_minus_1 = DenseMatrix.zeros[Double](out_dim, 1)
  var bottom_diff_x_t_minus_1 = DenseMatrix.zeros[Double](input_dim, 1)


  var cell_prev_t_minus_1 = DenseMatrix.zeros[Double](out_dim, 1)
  var h_prev_t_minus_1 = DenseMatrix.zeros[Double](out_dim, 1)

  var xt = DenseMatrix.zeros[Double](input_dim, 1)
  var xc = DenseMatrix.zeros[Double](input_dim + out_dim, 1)


  /**
    * forward propagation
    *
    * @param xt
    * @param cell_prev
    * @param h_prev
    */
  def forward(xt : DenseMatrix[Double], cell_prev : DenseMatrix[Double], h_prev: DenseMatrix[Double], param: LSTMLayerParam): Unit ={
    this.cell_prev_t_minus_1 = cell_prev
    this.h_prev_t_minus_1 = h_prev

    this.xt = xt
    this.xc = DenseMatrix.vertcat(this.xt, this.h_prev_t_minus_1)

    this.f = sigmoid( (param.Wf*this.xc).asInstanceOf[DenseMatrix[Double]] + param.Bf)
    this.i = sigmoid( (param.Wi*this.xc).asInstanceOf[DenseMatrix[Double]] + param.Bi)
    this.o = sigmoid( (param.Wo*this.xc).asInstanceOf[DenseMatrix[Double]] + param.Bo)
    this.g = tanh( (param.Wg*this.xc).asInstanceOf[DenseMatrix[Double]] + param.Bg )

    /**
      * :* means element wise
      */
    this.state_cell = this.g :*  this.i + this.cell_prev_t_minus_1 :* this.f
    this.state_h = this.o :*  tanh( this.state_cell )

  }

  /**
    * backward propagation
    *
    * @param top_diff_H_t all lose after time t, dH(t) = dh(t) +  dH(t+1)
    * @param top_diff_cell_t_plus_1 cell loss at t+1
    */
  def backward(top_diff_H_t : DenseMatrix[Double], top_diff_cell_t_plus_1:DenseMatrix[Double], param: LSTMLayerParam): Unit = {

    this.diff_cell_t = this.o :* ((1.0 - this.g :* this.g) :* this.g) :* top_diff_H_t + top_diff_cell_t_plus_1

    val diff_o = tanh( this.state_cell ) :* top_diff_H_t
    val diff_f = this.cell_prev_t_minus_1 :*  this.diff_cell_t
    val diff_i = this.g :* this.diff_cell_t
    val diff_g = this.i :* this.diff_cell_t

    /**
      * diffs w.r.t. vector inside sigma / tanh function
      */
    val do_input = (1.0 - this.o) :* this.o :* diff_o
    val df_input = (1.0 - this.f) :* this.f :* diff_f
    val di_input = (1.0 - this.i) :* this.i :* diff_i
    val dg_input = (1.0 - this.g :* this.g) :* diff_g

    /**
      * diffs w.r.t. inputs
      */

    param.wi_diff += LSTM.concatDiff(di_input, this.xt, this.h_prev_t_minus_1)
    param.wf_diff += LSTM.concatDiff(df_input, this.xt, this.h_prev_t_minus_1)
    param.wo_diff += LSTM.concatDiff(do_input, this.xt, this.h_prev_t_minus_1)
    param.wg_diff += LSTM.concatDiff(dg_input, this.xt, this.h_prev_t_minus_1)
    param.bi_diff += di_input
    param.bf_diff += df_input
    param.bo_diff += do_input
    param.bg_diff += dg_input


    var dxc = DenseMatrix.zeros[Double](param.concat_len, 1)
    dxc += param.Wo.t * do_input
    dxc += param.Wf.t * df_input
    dxc += param.Wi.t * di_input
    dxc += param.Wg.t * dg_input

    bottom_diff_h_t_minus_1 = this.diff_cell_t :* this.f
    bottom_diff_cell_t_minus_1 = dxc(param.input_dim to -1, ::)
    bottom_diff_x_t_minus_1 = dxc( 0 to param.input_dim, ::)
  }

}

/**
  * good paper: http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
  */
class LossLayer extends  Serializable {
  def negative_log_likelihood(label: DenseMatrix[Double], pred: DenseMatrix[Double]): Double = {
    0.0
  }

  def diff(label: DenseMatrix[Double], pred: DenseMatrix[Double]): DenseMatrix[Double] = {
    null
  }
}

class  softMaxLossLayer extends LossLayer{
  override def negative_log_likelihood(label: DenseMatrix[Double], pred: DenseMatrix[Double]): Double = {
    /*
    println("pred")
    println(pred)
    println("labels")
    println(label)
    */
    val pre_exp = exp(pred)
    val pre = argmax(pre_exp)
    val pre_label = label( pre )


    scala.math.log( sum( pre_exp ) ) - pre_label
  }

  override  def diff(label: DenseMatrix[Double], pred: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pre_exp = exp(pred)
    val softmax_pre = ( pre_exp/sum(pre_exp) ).asInstanceOf[ DenseMatrix[Double] ]

    softmax_pre - label
  }
}

class  simpleLossLayer extends LossLayer{
  override  def negative_log_likelihood(label: DenseMatrix[Double], pred: DenseMatrix[Double]): Double = {
    /*
    println("pred")
    println(pred)
    println("labels")
    println(label)
    */
    val diff  =   pred - label

    sum(diff :* diff)
  }

  override  def diff(label: DenseMatrix[Double], pred: DenseMatrix[Double]): DenseMatrix[Double] = {

     (pred - label):*= 2.0
  }
}


class LstmNeuralNetwork(val input_dim: Int, val hidden_dims : Seq[Int],val layer_size: Int =1, val lossLayer : LossLayer){
  assert(this.hidden_dims.length == this.layer_size && layer_size>=1)

  val LstmParams = new ArrayBuffer[LSTMLayerParam]()
  val y_out = new ArrayBuffer[ DenseMatrix[Double] ]()


  LstmParams.append(LSTMLayerParam(this.input_dim, hidden_dims.head))
  for(idx <- 1 until hidden_dims.length){
    LstmParams.append( LSTMLayerParam(hidden_dims.apply(idx - 1 ), hidden_dims.apply(idx)))
  }

  def forward_propagation(x_input:Seq[ DenseMatrix[Double] ]): Seq[ LSTMLayerNode ] = {
    this.y_out.clear()
    val nodes = new ArrayBuffer[ LSTMLayerNode ]()

    val first_node = new LSTMLayerNode(input_dim, hidden_dims.head)
    first_node.forward(x_input.head, DenseMatrix.zeros[Double](hidden_dims.head, 1), DenseMatrix.zeros[Double](hidden_dims.head, 1), LstmParams.head)
    nodes.append( first_node )
    this.y_out.append( first_node.state_h )


    for(idx<- 1 until x_input.size){
      val cell_pre = nodes.apply( idx -1 ).state_cell
      val h_pre = nodes.apply( idx -1 ).state_h
      val cur_node = new LSTMLayerNode(input_dim, hidden_dims.head)
      cur_node.forward(x_input.apply(idx), cell_pre, h_pre, LstmParams.head)
      nodes.append( cur_node )
      this.y_out.append( cur_node.state_h )
    }

    nodes
  }

  def backward_propagation(x_input:Seq[ DenseMatrix[Double]],
                           labels : Seq[ DenseMatrix[Double]]): Double = {
    val nodes = this.forward_propagation(x_input)

    val last_node = x_input.length - 1

    assert(x_input.length == nodes.length)

    var loss = lossLayer.negative_log_likelihood(labels.apply( last_node ), nodes.apply( last_node ).state_h)
    val diff_h = lossLayer.diff(labels.apply( last_node ), nodes.apply( last_node ).state_h)
    val diff_cell = DenseMatrix.zeros[Double](hidden_dims.head, 1)

    nodes.apply( last_node ).backward(diff_h, diff_cell, LstmParams.head)


    for(idx<- (0 until nodes.length-1).reverse ) {
      loss += lossLayer.negative_log_likelihood(labels.apply( idx ), nodes.apply( idx ).state_h)

      var diff_h = lossLayer.diff(labels.apply( idx ), nodes.apply(idx).state_h)
      diff_h +=  nodes.apply( idx+1 ).bottom_diff_h_t_minus_1
      val diff_cell = nodes.apply( idx+1 ).bottom_diff_cell_t_minus_1

      nodes.apply( idx ).backward(diff_h, diff_cell, LstmParams.head)
    }

    loss
  }

}


object LSTM{
  def concatDiff(diff: DenseMatrix[Double], xt: DenseMatrix[Double], ht_minus_1: DenseMatrix[Double]) : DenseMatrix[Double] = {
    val wi_diff_x = (diff * xt.t).asInstanceOf[DenseMatrix[Double]]
    val wi_diff_h = (diff * ht_minus_1.t).asInstanceOf[DenseMatrix[Double]]
    DenseMatrix.horzcat(wi_diff_x, wi_diff_h)
  }
  /*
  implicit def +=(x : DenseMatrix[Double],y: DenseMatrix[Double]): DenseMatrix[Double] = {
    (x + y).asInstanceOf[DenseMatrix[Double]]
  }
  */

  def main(args: Array[String]) {
    val data = Seq(
      DenseMatrix.rand[Double](5,1),
      DenseMatrix.rand[Double](5,1),
      DenseMatrix.rand[Double](5,1),
      DenseMatrix.rand[Double](5,1),
      DenseMatrix.rand[Double](5,1),
      DenseMatrix.rand[Double](5,1)
    )

    val labels = Seq(
      DenseMatrix((0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0)).t,
      DenseMatrix((0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0)).t,
      DenseMatrix((0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0)).t,
      DenseMatrix((0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0)).t,
      DenseMatrix((0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0)).t,
      DenseMatrix((0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0)).t
    )

    val simpleLSTM = new LstmNeuralNetwork(5, Seq(8), 1, new simpleLossLayer)

    simpleLSTM.forward_propagation(data)

    for(idx<- 0 to 30000){
      val loss_new = simpleLSTM.backward_propagation(data, labels)
      println("loss: " + loss_new.toString)
      simpleLSTM.LstmParams.head.update_param(0.02)
    }

    val out = simpleLSTM.y_out
    for(idx<- out.indices){
      val outnode  = out.apply(idx)
      println( outnode )
      println("------")
      val pre = DenseMatrix.zeros[Double](outnode.rows, outnode.cols)
      pre( argmax(outnode) ) = 1.0
      println( pre )
      println("------")
    }

  }
}
