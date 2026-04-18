Variance-stratified (residual) FDR
==================================

Kompot's default test statistic is the per-gene Mahalanobis distance

.. math::

   D^2_g = (\mu^{(A)}_g - \mu^{(B)}_g)^\top
           (\Sigma^{(A)}_g + \Sigma^{(B)}_g)^{-1}
           (\mu^{(A)}_g - \mu^{(B)}_g)

where :math:`\mu` and :math:`\Sigma` are the GP posterior mean and
covariance over the manifold.  When
``GPSettings.use_empirical_variance=False`` (the default — chosen so
per-cell sampling noise is not conflated with epistemic posterior
uncertainty), :math:`D^2_g` scales monotonically with per-gene
expression variance :math:`\sigma^2_g`.  The gene-shuffled empirical
null inherits that scaling, so the 1-D local FDR is genome-wide
calibrated but not gene-specific.

This matters most in low-replication designs: with two biological
replicates per condition, every 2-vs-2 partition — including the real
one — produces a mean difference roughly proportional to
:math:`\sigma^2_g`, so any partition ranks high-variance genes at the
top.  Under a balanced permutation of the Tal1 chimera dataset the raw
Mahalanobis rank correlation between real and permuted partitions is
Spearman 0.99 genome-wide and 0.79 in the top 10 %.

When to enable
--------------

Enable variance-stratified FDR when any of the following apply:

* **n ≈ 2 biological replicates per condition** and you want to
  interpret the top DE list as *condition-specific* rather than
  *variability-driven*.
* You notice that the top-ranked DE genes are suspiciously similar
  between the real comparison and a balanced permutation of condition
  labels.
* You want the FDR calibration to be robust to per-gene mean and
  variance even when the manifold partition is unbalanced.

Keep the default (``mode="raw"``) when you have well-replicated data
(≥3 per condition) and want the classical behaviour.

Usage
-----

.. code-block:: python

   import kompot

   kompot.de(
       adata, "condition", "WT", "Mutant",
       fdr=kompot.FDRSettings(
           mode="variance_stratified",
           null_trend_features=("log_mean", "log_var"),
           null_trend_model="poly3",
       ),
   )

The raw Mahalanobis, raw local FDR, and ``is_de`` columns are still
written unchanged.  In addition, the following columns are added to
``adata.var``:

================================================  ========================================================
Column                                            Meaning
================================================  ========================================================
``..._residual_mahalanobis``                      :math:`\log(1 + D^2_g) - \hat\varphi(m_g, v_g)`
``..._residual_z``                                residual standardised by :math:`\hat\sigma_{\text{null}}`
``..._residual_local_fdr``                        local FDR on :math:`Z`
``..._residual_is_de``                            ``True`` at the configured threshold
``..._residual_pvalue`` *(additional stats)*      empirical p-value from the residual null
``..._residual_tail_fdr`` *(additional stats)*    tail FDR on :math:`Z`
``..._residual_log_mean`` *(additional stats)*    :math:`\log(1+\bar x_g)`
``..._residual_log_var`` *(additional stats)*     :math:`\log(1+\operatorname{Var}(x_g))`
================================================  ========================================================

Method
------

1. Compute :math:`m_g = \log(1 + \bar x_g)` and
   :math:`v_g = \log(1 + \operatorname{Var}(x_g))` per gene from the
   expression layer used for DE.
2. Fit the null-trend surface
   :math:`\hat\varphi(m, v) = \mathbb{E}[\log(1 + D^2_{\text{null}}) \mid m, v]`
   by ordinary least squares on the gene-shuffled null draws
   (``null_trend_model="poly3"`` is a 7-term tensor polynomial).
3. Estimate the homoscedastic null residual scale
   :math:`\hat\sigma_{\text{null}}` from the fit.
4. Define the residualised statistic
   :math:`R_g = \log(1 + D^2_g) - \hat\varphi(m_g, v_g)` and
   :math:`Z_g = R_g / \hat\sigma_{\text{null}}`.
5. Feed :math:`\max(0, Z_g)` to kompot's existing 1-D local FDR (the
   null of :math:`Z` is centred at zero by construction).  Below-null
   genes (negative :math:`Z`) are assigned local FDR ≈ 1.

What this does and does not do
------------------------------

**Does**

* Remove the strictly gene-level monotone dependence on per-gene mean
  and variance from the statistic before FDR calibration.
* Rank genes by how unusual their Mahalanobis is *conditional on*
  their mean and variance.

**Does not**

* Remove manifold / cell-type structure.  Genes that vary strongly
  between cell types will still have large Mahalanobis under any
  cell partition that splits cell-type mixtures.
* Fix the underlying experimental-design limitation of two
  biological replicates per condition.  With sample fully confounded
  with condition, no statistic can separate condition-level biology
  from embryo-to-embryo variability in principle.

Tal1 validation
---------------

On the Tal1 chimera dataset (2 mutant + 2 WT embryos, 29 453 genes),
residualisation flips the top-20 genes from an imprinting/X-linked
variability signature to the known Tal1 hematopoietic phenotype:

* **Raw top-20 hematopoietic genes:** 5 (Hbb-bh1, Hba-x, Hbb-y,
  Hba-a1, Hba-a2)
* **Residual top-20 hematopoietic genes:** 11 (also Itga2b, Lyl1,
  Stab1, Stab2, Tfec, Ackr1)
* **Raw top-20 imprinted genes:** 12
* **Residual top-20 imprinted genes:** 0

See the Tal1 validation report in the kompot manuscript supplement for
the full analysis.
