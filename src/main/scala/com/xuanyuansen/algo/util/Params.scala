package com.xuanyuansen.algo.util

/**
 * Created by wangshuai on 16/7/28.
 */

trait Params extends Serializable {

}

case class EmptyParams() extends Params {
  override def toString: String = "Empty"
}