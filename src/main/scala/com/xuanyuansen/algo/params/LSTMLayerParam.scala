package com.xuanyuansen.algo.params

import breeze.linalg.DenseMatrix

/**
 * Created by wangshuai on 16/7/28.
 */
class LSTMLayerParam extends LayerParam {
  var concat_len = 0

  var Wo: DenseMatrix[Double] = null
  var Wf: DenseMatrix[Double] = null
  var Wi: DenseMatrix[Double] = null
  var Wg: DenseMatrix[Double] = null

  var Bo: DenseMatrix[Double] = null
  var Bf: DenseMatrix[Double] = null
  var Bi: DenseMatrix[Double] = null
  var Bg: DenseMatrix[Double] = null

  var wo_diff: DenseMatrix[Double] = null
  var wf_diff: DenseMatrix[Double] = null
  var wi_diff: DenseMatrix[Double] = null
  var wg_diff: DenseMatrix[Double] = null

  var bo_diff: DenseMatrix[Double] = null
  var bf_diff: DenseMatrix[Double] = null
  var bi_diff: DenseMatrix[Double] = null
  var bg_diff: DenseMatrix[Double] = null

  override def initParam(in_dim: Int, out_dim: Int): Unit = {
    this.concat_len = in_dim + out_dim
    this.logger.info("concat size is %d".format(this.concat_len))
    /**
     * f: forget gate
     * g: cell gate
     * o: output gate
     * i: input gate
     */
    this.Wo = DenseMatrix.rand[Double](out_dim, this.concat_len)
    this.Wf = DenseMatrix.rand[Double](out_dim, this.concat_len)
    this.Wi = DenseMatrix.rand[Double](out_dim, this.concat_len)
    this.Wg = DenseMatrix.rand[Double](out_dim, this.concat_len)

    this.Bo = DenseMatrix.rand[Double](out_dim, 1)
    this.Bf = DenseMatrix.rand[Double](out_dim, 1)
    this.Bi = DenseMatrix.rand[Double](out_dim, 1)
    this.Bg = DenseMatrix.rand[Double](out_dim, 1)

    /**
     *
     */
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
