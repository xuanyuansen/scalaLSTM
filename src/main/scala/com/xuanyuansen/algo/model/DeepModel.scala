package com.xuanyuansen.algo.model

import com.xuanyuansen.algo.layers.Layer

/**
 * Created by wangshuai on 16/7/28.
 * core object of deep learning model
 */
class DeepModel {
  def save_model(): Unit = {}

  def load_model(): Unit = {}

  def fit(): Unit = {}

  def fit_mini_batch(): Unit = {}

  def fit_sgd(): Unit = {}

  def predict(): Unit = {}

  def predict_prob(): Unit = {}

  def evaluate(): Unit = {}
}

class RecurrentModel extends DeepModel {
  val layers: Seq[Layer] = null
}
