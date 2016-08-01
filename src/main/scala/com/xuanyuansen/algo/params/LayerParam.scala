package com.xuanyuansen.algo.params

import com.typesafe.scalalogging.slf4j.Logger
import org.slf4j.LoggerFactory

/**
  * Created by wangshuai on 16/7/28.
  */

class LayerParam extends Serializable{
  @transient lazy protected val logger = Logger(LoggerFactory.getLogger(this.getClass))

  protected var in_dim: Int = 0
  protected var out_dim: Int = 0

  def shape() : (Int, Int) = (in_dim, out_dim)

  def initParam(input_dim : Int,output_dim: Int) : Unit = {}

}

