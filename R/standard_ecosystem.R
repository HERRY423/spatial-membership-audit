# Out-of-the-Box R / Giotto Standard Ecosystem & Operational Action Guide
# Additive companion to R/ecosystem.R for spatial_membership_audit (v0.3.0+).
# Provides:
#   1. Declarative custom marker-rule probability evaluation in R
#   2. Exact pair availability/contact moments; conditional ratio CV is a Delta approximation
#   3. Biologist's Operational Decision Guide & 11-regime triage engine in R
#   4. Seamless Giotto object extraction and end-to-end audit wrapper

sma_binom_tail_r <- function(k, m, p) {
  if (length(p) != 1L || is.na(p) || p < 0 || p > 1) {
    stop("Retention probability p must be a single numeric in [0, 1].")
  }
  if (m <= 0L) {
    return(rep(1.0, length(k)))
  }
  k_int <- as.integer(k)
  if (p <= 0) {
    return(as.numeric(k_int >= m & FALSE))
  }
  if (p >= 1) {
    return(as.numeric(k_int >= m))
  }
  out <- numeric(length(k_int))
  ok <- k_int >= m
  if (any(ok)) {
    # P(Binom(k, p) >= m) = pbinom(m - 1, size = k, prob = p, lower.tail = FALSE)
    out[ok] <- stats::pbinom(m - 1L, size = k_int[ok], prob = p, lower.tail = FALSE)
  }
  out
}

# Joint capped-count enumeration preserves dependence when predicates share genes.
# No multiplication of overlapping event probabilities and no silent MC fallback.
sma_rule_states_r <- function(predicates, genes, max_states = 65536L) {
  caps <- stats::setNames(rep(0, length(genes)), genes)
  threshold <- function(x, default) {
    if (is.null(x)) x <- default
    if (length(x) != 1L || !is.finite(x) || x < 0 || x != floor(x))
      stop("Rule thresholds must be finite non-negative integers.")
    x
  }
  visit <- function(pred) {
    if (!is.list(pred) || length(pred$type) != 1L) stop("Each predicate needs one type.")
    tp <- pred$type
    if (tp %in% c("gene", "gene_sum")) {
      gs <- if (tp == "gene") as.character(pred$gene) else as.character(pred$genes)
      if (!length(gs) || anyNA(gs) || anyDuplicated(gs) || any(!gs %in% genes))
        stop("Predicate genes must be unique and present in counts.")
      if (tp == "gene" && length(gs) != 1L) stop("gene predicate needs one gene.")
      cap <- threshold(if (tp == "gene") pred$min_count else pred$min_sum, 1L)
      caps[gs] <<- pmax(caps[gs], cap)
    } else if (tp %in% c("all_of", "any_of")) {
      if (!is.list(pred$items) || !length(pred$items)) stop("Compound predicates need items.")
      invisible(lapply(pred$items, visit))
    } else if (tp == "not") visit(pred$item)
    else stop(sprintf("Unsupported predicate type '%s'.", tp))
  }
  invisible(lapply(predicates, visit))
  caps <- caps[caps > 0]
  if (length(max_states) != 1L || !is.finite(max_states) || max_states < 1 ||
      sum(log(as.double(caps)+1)) > log(max_states)+1e-12)
    stop("Exact joint rule state budget exceeded; simplify the rule or explicitly raise max_states.")
  states <- if (length(caps)) as.matrix(expand.grid(lapply(caps, function(k) 0:k))) else matrix(numeric(), 1L, 0L)
  colnames(states) <- names(caps)
  evaluate <- function(pred) {
    tp <- pred$type
    if (tp == "gene") {
      k <- threshold(pred$min_count, 1L)
      return(if (k == 0) rep(TRUE, nrow(states)) else states[, pred$gene] >= k)
    }
    if (tp == "gene_sum") {
      k <- threshold(pred$min_sum, 1L)
      return(if (k == 0) rep(TRUE, nrow(states)) else rowSums(states[, pred$genes, drop=FALSE]) >= k)
    }
    if (tp == "not") return(!evaluate(pred$item))
    vals <- lapply(pred$items, evaluate)
    Reduce(if (tp == "all_of") `&` else `|`, vals)
  }
  list(states=states, caps=caps, truth=lapply(predicates, evaluate))
}

sma_joint_rule_probabilities_r <- function(counts, genes, p, truth, design) {
  if (length(p) != 1L || !is.finite(p) || p < 0 || p > 1) stop("p must be in [0,1].")
  states <- design$states
  out <- matrix(0, nrow(counts), ncol(truth))
  cache <- new.env(hash=TRUE, parent=emptyenv())
  for (i in seq_len(nrow(counts))) {
    vals <- counts[i, match(names(design$caps), genes)]
    key <- paste0("k", paste(vals, collapse=","))
    if (!exists(key, envir=cache, inherits=FALSE)) {
      mass <- rep(1, nrow(states))
      for (g in seq_along(design$caps)) {
        cap <- design$caps[g]
        pr <- c(stats::dbinom(0:(cap-1), size=vals[g], prob=p),
                stats::pbinom(cap-1, size=vals[g], prob=p, lower.tail=FALSE))
        mass <- mass * pr[states[, g]+1L]
      }
      assign(key, as.numeric(crossprod(mass, truth)), envir=cache)
    }
    out[i, ] <- get(key, envir=cache, inherits=FALSE)
  }
  out
}

sma_eval_predicate_r <- function(pred, counts, gene_index, p) {
  genes <- names(gene_index)
  design <- sma_rule_states_r(list(pred), genes)
  sma_joint_rule_probabilities_r(as.matrix(counts), genes, p,
                               matrix(design$truth[[1]], ncol=1), design)[, 1]
}

sma_evaluate_custom_rule_r <- function(counts, genes, eligible=NULL, p=1.0,
                                      rule_spec, max_states=65536L) {
  counts <- as.matrix(counts)
  genes <- as.character(genes)
  if (ncol(counts) != length(genes) || anyNA(genes) || anyDuplicated(genes))
    stop("Unique gene names must match count columns.")
  if (any(!is.finite(counts)) || any(counts < 0) || any(counts != floor(counts)))
    stop("counts must be finite non-negative integers.")
  if (is.null(eligible)) eligible <- rep(TRUE, nrow(counts))
  if (!is.logical(eligible) || length(eligible) != nrow(counts) || anyNA(eligible))
    stop("eligible must be an aligned fixed boolean mask without NA.")
  classes <- rule_spec$classes
  if (!is.list(classes) || !length(classes)) stop("rule_spec$classes must be non-empty.")
  predicates <- lapply(classes, function(cl) {
    if (is.null(cl$veto)) cl$positive else
      list(type="all_of", items=list(cl$positive, list(type="not", item=cl$veto)))
  })
  design <- sma_rule_states_r(predicates, genes, max_states)
  truth <- do.call(cbind, design$truth)
  if (any(rowSums(truth) > 1L)) stop("Candidate classes overlap in the joint rule state space.")
  unassigned <- if (is.null(rule_spec$unassigned_name)) "UNASSIGNED" else as.character(rule_spec$unassigned_name)
  names_out <- c(unassigned, vapply(classes, function(cl) as.character(cl$name), character(1)))
  if (anyNA(names_out) || any(!nzchar(names_out)) || anyDuplicated(names_out))
    stop("Class names must be distinct non-empty strings.")
  assigned <- sma_joint_rule_probabilities_r(counts, genes, p, truth, design)
  assigned[!eligible, ] <- 0
  probs <- cbind(pmax(0, 1-rowSums(assigned)), assigned)
  colnames(probs) <- names_out
  attr(probs, "method") <- "exact_joint_capped_binomial_states"
  attr(probs, "joint_states") <- nrow(design$states)
  probs
}

sma_stromal_myeloid_rule_spec_r <- function() {
  list(
    name = "stromal_myeloid_4gene_v1",
    unassigned_name = "UNASSIGNED",
    classes = list(
      list(
        name = "stromal",
        positive = list(
          type = "any_of",
          items = list(
            list(type = "gene", gene = "LUM", min_count = 1L),
            list(type = "gene", gene = "POSTN", min_count = 1L)
          )
        ),
        veto = list(
          type = "any_of",
          items = list(
            list(type = "gene", gene = "ITGAM", min_count = 1L),
            list(type = "gene", gene = "CD14", min_count = 1L)
          )
        )
      ),
      list(
        name = "myeloid",
        positive = list(
          type = "any_of",
          items = list(
            list(type = "gene", gene = "ITGAM", min_count = 1L),
            list(type = "gene", gene = "CD14", min_count = 1L)
          )
        ),
        veto = list(
          type = "any_of",
          items = list(
            list(type = "gene", gene = "LUM", min_count = 1L),
            list(type = "gene", gene = "POSTN", min_count = 1L)
          )
        )
      )
    )
  )
}

sma_pair_availability_and_moments_r <- function(probabilities, adjacency, col_a=2L, col_b=3L, tol=1e-12) {
  P <- as.matrix(probabilities)
  W <- as.matrix(adjacency)
  n <- nrow(P)
  if (any(!is.finite(P)) || any(P < 0) || any(P > 1) || any(rowSums(P) > 1+tol))
    stop("Probabilities must be finite categorical row probabilities, with sum <= 1.")
  if (!identical(dim(W), c(n,n)) || any(!is.finite(W)) || any(!(W %in% c(0,1))) ||
      !isTRUE(all.equal(W,t(W),check.attributes=FALSE)) || any(diag(W) != 0))
    stop("adjacency must be a symmetric, hollow, binary N x N graph.")
  if (is.character(col_a)) col_a <- match(col_a,colnames(P))
  if (is.character(col_b)) col_b <- match(col_b,colnames(P))
  cs <- c(col_a,col_b)
  if (length(cs) != 2L || anyNA(cs) || any(cs != floor(cs)) ||
      any(cs < 1 | cs > ncol(P)) || col_a == col_b) stop("Two distinct valid class columns required.")
  pa <- P[,col_a]; pb <- P[,col_b]
  if (any(pa+pb > 1)) stop("Selected class probabilities exceed one; repair source roundoff explicitly.")

  # Four presence states: neither, A only, B only, both. Each node has ONE label.
  # P(A_i and B_i) is zero, not pa[i]*pb[i]: both classes require distinct nodes.
  # This evaluates 1-prod(1-pa)-prod(1-pb)+prod(1-pa-pb) without cancellation.
  # Propagate joint mass and raw count moments without subtracting near-one products.
  dp <- matrix(0,4,6,dimnames=list(NULL,c("mass","a","b","aa","bb","ab")))
  dp[1,1] <- 1
  destinations <- list(1:4,c(2,2,4,4),c(3,4,3,4))
  for (i in seq_len(n)) {
    next_dp <- matrix(0,4,6)
    for (label in 1:3) {
      w <- c(1-pa[i]-pb[i],pa[i],pb[i])[label]
      if (w == 0) next
      da <- as.numeric(label == 2); db <- as.numeric(label == 3)
      for (s in 1:4) {
        z <- dp[s,]; dest <- destinations[[label]][s]
        add <- c(z[1],z[2]+da*z[1],z[3]+db*z[1],
                 z[4]+2*da*z[2]+da*z[1], z[5]+2*db*z[3]+db*z[1],
                 z[6]+da*z[3]+db*z[2])
        next_dp[dest,] <- next_dp[dest,]+w*add
      }
    }
    dp <- next_dp
  }
  pp <- dp[4,1]
  possible_coexistence <- sum(pa>0)*sum(pb>0)-sum(pa>0 & pb>0) > 0
  probability_underflow <- pp == 0 && possible_coexistence
  edges <- which(upper.tri(W) & W > 0, arr.ind=TRUE)
  m <- nrow(edges); u <- edges[,1]; v <- edges[,2]
  ab <- pa[u]*pb[v]; ba <- pb[u]*pa[v]
  contact_underflow <- any((pa[u]>0 & pb[v]>0 & ab==0) | (pb[u]>0 & pa[v]>0 & ba==0))
  qa <- ab+ba
  mc <- sum(qa); ma <- sum(pa); mb <- sum(pb)
  if (contact_underflow) {
    # A positive contact probability rounded to zero. Dividing that rounded zero
    # by a small P_plus can hide a nonzero conditional mean: return unresolved.
    return(list(P_plus=if (probability_underflow) NA_real_ else pp,
      P_plus_computed=pp,P_plus_underflow=probability_underflow,contact_product_underflow=TRUE,
      P_00=dp[1,1],P_pos_0=dp[2,1],P_0_pos=dp[3,1],
      mean_C=NA_real_,mean_C_computed=mc,var_C=NA_real_,cond_mean_C=NA_real_,
      cond_cv_C=NA_real_,cond_cv_ratio=NA_real_,mean_Na=ma,mean_Nb=mb,
      cond_mean_Na=NA_real_,cond_mean_Nb=NA_real_,conditional_covariance=matrix(NA_real_,3,3),
      n_edges=m,status="NUMERICAL_UNDERFLOW",
      ratio_cv_method="first_order_delta_using_exact_conditional_joint_moments",ratio_cv_is_exact=FALSE))
  }
  sa <- as.numeric(W %*% pa); sb <- as.numeric(W %*% pb)
  shared <- sum(pa*(1-pa)*(sb^2-as.numeric(W %*% pb^2)) +
                pb*(1-pb)*(sa^2-as.numeric(W %*% pa^2)) -
                2*pa*pb*(sa*sb-as.numeric(W %*% (pa*pb))))
  vc <- sum(qa*(1-qa))+shared
  if (vc < -tol*max(1,mc^2)) stop("Negative contact variance beyond rounding tolerance.")
  vc <- max(0,vc)
  mu <- rep(NA_real_,3); sigma <- matrix(NA_real_,3,3)
  cv_c <- cv_ratio <- NA_real_
  status <- if (probability_underflow) "NUMERICAL_UNDERFLOW" else if (pp == 0) "NO_FINITE_RATIO" else if (m == 0) "NO_GRAPH_EDGES" else
    if (mc == 0) "ZERO_NUMERATOR_CV_UNDEFINED" else "CONDITIONAL_DELTA_APPROXIMATION"
  if (pp > 0) {
    mu <- c(mc,dp[4,2],dp[4,3])/pp
    eca <- sum(qa*(1+ma-pa[u]-pa[v]))/pp
    ecb <- sum(qa*(1+mb-pb[u]-pb[v]))/pp
    second <- matrix(c((vc+mc^2)/pp,eca,ecb,
                       eca,dp[4,4]/pp,dp[4,6]/pp,
                       ecb,dp[4,6]/pp,dp[4,5]/pp),3,3)
    sigma <- second-tcrossprod(mu)
    if (mc > 0 && m > 0) {
      cv_c <- sqrt(max(0,sigma[1,1]))/mu[1]
      grad <- c(1/mu[1],-1/mu[2],-1/mu[3])
      relvar <- as.numeric(t(grad) %*% sigma %*% grad)
      if (relvar < -tol*max(1,sum(abs(outer(grad,grad)*sigma))))
        stop("Conditional delta variance negative beyond rounding tolerance.")
      cv_ratio <- sqrt(max(0,relvar))
    }
  }
  list(P_plus=if (probability_underflow) NA_real_ else pp,
       P_plus_computed=pp,P_plus_underflow=probability_underflow,
       contact_product_underflow=FALSE,
       P_00=dp[1,1],P_pos_0=dp[2,1],P_0_pos=dp[3,1],
       mean_C=mc,var_C=vc,cond_mean_C=mu[1],cond_cv_C=cv_c,cond_cv_ratio=cv_ratio,
       mean_Na=ma,mean_Nb=mb,cond_mean_Na=mu[2],cond_mean_Nb=mu[3],
       conditional_covariance=sigma,n_edges=m,status=status,
       ratio_cv_method="first_order_delta_using_exact_conditional_joint_moments",
       ratio_cv_is_exact=FALSE)
}

sma_triage_record_r <- function(p_plus, cond_cv, cond_mean_contact=NA_real_,
  mean_na=NA_real_, mean_nb=NA_real_, n_edges=NA_integer_,
  has_non_monotonic_retention=FALSE, p_plus_high=.95, p_plus_low=.50,
  p_plus_absent_tol=1e-12, cv_stable=.25, cv_moderate=.50, cv_severe=1.00) {
  # p_plus_absent_tol is retained for API compatibility; positive values are NEVER structural zeros.
  finite <- function(v) length(v)==1L && !is.na(v) && is.finite(v)
  if (!(0<p_plus_low && p_plus_low<p_plus_high && p_plus_high<=1 &&
        0<cv_stable && cv_stable<cv_moderate && cv_moderate<=cv_severe))
    stop("Invalid diagnostic thresholds.")
  if (finite(n_edges) && n_edges==0) code<-"STRUCTURAL_NO_GRAPH_EDGES"
  else if (!finite(p_plus) || p_plus<0 || p_plus>1) code<-"NUMERICAL_UNRESOLVED"
  else if (p_plus==0) code<-"STRUCTURAL_ABSENT_POPULATION"
  else if (finite(cond_mean_contact) && cond_mean_contact==0) code<-"DEFINED_ZERO_CONTACT"
  else if (!finite(cond_cv) || cond_cv<0) code<-"NUMERICAL_UNRESOLVED"
  else if (isTRUE(has_non_monotonic_retention)) code<-"NON_MONOTONIC_VETO_ENTRY"
  else if (p_plus<p_plus_low) code<-if (cond_cv<=cv_moderate) "TIER_5A_CONDITIONING_TRAP" else "TIER_5B_SEVERE_FRAGILITY"
  else if (p_plus<p_plus_high) code<-"TIER_4_AVAILABILITY_WARNING"
  else if (cond_cv<=cv_stable) code<-"TIER_1_ROBUST"
  else if (cond_cv<=cv_moderate) code<-"TIER_2_MODERATE_DISPERSION"
  else code<-"TIER_3_CONTACT_VOLATILE"
  actions <- list(
    STRUCTURAL_NO_GRAPH_EDGES=c("Verify coordinate units and the specified graph construction. With M=0 the ratio is undefined; retain NA.","\u6838\u5bf9\u5750\u6807\u5355\u4f4d\u4e0e\u9884\u8bbe\u5efa\u56fe\u65b9\u5f0f\uff1bM=0 \u65f6\u6bd4\u503c\u4e0d\u53ef\u5b9a\u4e49\uff0c\u4fdd\u7559 NA\u3002"),
    STRUCTURAL_ABSENT_POPULATION=c("The two exclusive classes cannot coexist under this specified model. Preserve NA; inspect marker coverage and class support.","\u5728\u6307\u5b9a\u4e92\u65a5\u5206\u7c7b\u6a21\u578b\u4e0b\uff0c\u4e24\u7c7b\u4e0d\u80fd\u540c\u65f6\u51fa\u73b0\uff1b\u4fdd\u7559 NA\uff0c\u6838\u5bf9 marker \u8986\u76d6\u548c\u7c7b\u522b\u652f\u6301\u3002"),
    NON_MONOTONIC_VETO_ENTRY=c("Inspect the complete retention curve and rule transitions. Veto dropout is one possible explanation; the trajectory alone does not establish it.","\u5c55\u793a\u5b8c\u6574\u4fdd\u7559\u7387\u66f2\u7ebf\u5e76\u6838\u67e5\u5206\u7c7b\u89c4\u5219\u53d8\u5316\uff1b\u5426\u51b3\u57fa\u56e0\u8131\u843d\u662f\u4e00\u79cd\u5f85\u6838\u5bf9\u89e3\u91ca\uff0c\u975e\u5355\u8c03\u8f68\u8ff9\u672c\u8eab\u4e0d\u80fd\u8bc1\u660e\u539f\u56e0\u3002"),
    DEFINED_ZERO_CONTACT=c("Under the fixed graph and label model, conditional contact is zero and relative CV is undefined. Report R=0 with P+; biological repulsion requires a separate null comparison.","\u5728\u56fa\u5b9a\u56fe\u548c\u6807\u7b7e\u6a21\u578b\u4e0b\u6761\u4ef6\u63a5\u89e6\u4e3a\u96f6\uff0c\u76f8\u5bf9 CV \u4e0d\u53ef\u5b9a\u4e49\uff1b\u5e76\u5217\u62a5\u544a R=0 \u4e0e P+\uff0c\u751f\u7269\u5b66\u6392\u65a5\u987b\u53e6\u6709\u96f6\u5047\u8bbe\u6bd4\u8f83\u3002"),
    NUMERICAL_UNRESOLVED=c("Preserve unresolved or missing values. Check inputs, numerical status and declared budget; label any MC/Sobol estimate and its sampling uncertainty.","\u4fdd\u7559\u672a\u89e3\u51b3\u6216\u7f3a\u5931\u72b6\u6001\uff1b\u6838\u5bf9\u8f93\u5165\u3001\u6570\u503c\u72b6\u6001\u4e0e\u9884\u7b97\uff0c\u82e5\u4f7f\u7528 MC/Sobol\uff0c\u6ce8\u660e\u8fd1\u4f3c\u65b9\u6cd5\u53ca\u62bd\u6837\u4e0d\u786e\u5b9a\u6027\u3002"),
    TIER_1_ROBUST=c("High availability and low conditional dispersion under the specified retention model. Report P+, CV and method; this diagnostic does not validate biological interpretation or cross-sample exchangeability.","\u5728\u6307\u5b9a\u4fdd\u7559\u6a21\u578b\u4e0b\u53ef\u7528\u7387\u9ad8\u3001\u6761\u4ef6\u79bb\u6563\u5ea6\u4f4e\uff1b\u62a5\u544a P+\u3001CV \u53ca\u8ba1\u7b97\u65b9\u6cd5\uff0c\u8fd9\u4e00\u8bca\u65ad\u4e0d\u9a8c\u8bc1\u751f\u7269\u5b66\u89e3\u91ca\u6216\u8de8\u6837\u672c\u53ef\u6bd4\u6027\u3002"),
    TIER_2_MODERATE_DISPERSION=c("Report availability and conditional dispersion; include the conditional mean when relevant. Numerical enclosures and biological sampling intervals answer different questions.","\u5e76\u5217\u62a5\u544a\u53ef\u7528\u7387\u3001\u6761\u4ef6\u79bb\u6563\u5ea6\u53ca\u5fc5\u8981\u7684\u6761\u4ef6\u5747\u503c\uff1b\u6570\u503c\u5305\u56f4\u533a\u95f4\u4e0e\u751f\u7269\u5b66\u62bd\u6837\u533a\u95f4\u56de\u7b54\u4e0d\u540c\u95ee\u9898\u3002"),
    TIER_3_CONTACT_VOLATILE=c("High conditional dispersion warrants inspecting contact counts and rule-boundary cells. CV alone identifies neither the cause nor the number of influential cells.","\u6761\u4ef6\u79bb\u6563\u5ea6\u8f83\u9ad8\uff0c\u53ef\u6838\u67e5\u63a5\u89e6\u8ba1\u6570\u4e0e\u95e8\u63a7\u8fb9\u754c\u7ec6\u80de\uff1b\u4ec5\u51ed CV \u4e0d\u80fd\u786e\u5b9a\u539f\u56e0\u6216\u5173\u952e\u7ec6\u80de\u6570\u91cf\u3002"),
    TIER_4_AVAILABILITY_WARNING=c("Report P+ alongside conditional CV. Treat changes to ROI size, radius or thresholds as explicitly labelled sensitivity analyses with a potentially different estimand.","\u5e76\u5217\u62a5\u544a P+ \u4e0e\u6761\u4ef6 CV\uff1b\u6539\u53d8 ROI\u3001\u534a\u5f84\u6216\u95e8\u69db\u987b\u660e\u786e\u6807\u4e3a\u654f\u611f\u6027\u5206\u6790\uff0c\u5e76\u8bf4\u660e\u76ee\u6807\u91cf\u53ef\u80fd\u6539\u53d8\u3002"),
    TIER_5A_CONDITIONING_TRAP=c("A low conditional CV can coexist with frequent undefined ratios. Retain both summaries; low CV alone is insufficient evidence of availability or biological robustness.","\u4f4e\u6761\u4ef6 CV \u53ef\u4ee5\u4e0e\u9891\u7e41\u4e0d\u53ef\u5b9a\u4e49\u540c\u65f6\u51fa\u73b0\uff1b\u4fdd\u7559\u4e24\u9879\u7ed3\u679c\uff0c\u4e0d\u80fd\u4ec5\u51ed\u4f4e CV \u63a8\u65ad\u9ad8\u53ef\u7528\u6027\u6216\u751f\u7269\u5b66\u7a33\u5065\u6027\u3002"),
    TIER_5B_SEVERE_FRAGILITY=c("Report frequent undefined ratios and high conditional dispersion. Do not infer biological absence from thinning dropout; any pooling or exclusion rule needs separate validation.","\u62a5\u544a\u9891\u7e41\u4e0d\u53ef\u5b9a\u4e49\u4e0e\u9ad8\u6761\u4ef6\u79bb\u6563\u5ea6\uff1b\u4e0d\u80fd\u5c06\u62bd\u7a00\u5931\u5b9a\u4e49\u63a8\u4f5c\u751f\u7269\u5b66\u7f3a\u5931\uff0c\u5408\u5e76\u6216\u5254\u9664\u89c4\u5219\u987b\u53e6\u884c\u9a8c\u8bc1\u3002")
  )
  rank<-switch(code,TIER_1_ROBUST=1L,DEFINED_ZERO_CONTACT=1L,TIER_2_MODERATE_DISPERSION=2L,
               TIER_3_CONTACT_VOLATILE=3L,TIER_4_AVAILABILITY_WARNING=3L,NON_MONOTONIC_VETO_ENTRY=3L,
               TIER_5A_CONDITIONING_TRAP=4L,5L)
  list(regime_code=code,severity_rank=rank,
       is_conditioning_trap=finite(p_plus) && finite(cond_cv) && p_plus>0 &&
         ((p_plus<p_plus_low && cond_cv<=cv_moderate) || (p_plus<p_plus_high && cond_cv<=cv_stable)),
       recommended_plot_encoding=tolower(code),action_en=actions[[code]][1],action_zh=actions[[code]][2],
       inferential_guardrail="Diagnostic conventions only; not validated filters, p values, FWER control or biological evidence. R ratio CV uses a conditional Delta approximation.")
}

sma_triage_table_r <- function(df, p_plus_col = "P_plus", cv_col = "cond_cv_ratio", cond_mean_col = "cond_mean_C") {
  n <- nrow(df)
  res <- lapply(seq_len(n), function(i) {
    p_val <- df[[p_plus_col]][i]
    cv_val <- df[[cv_col]][i]
    cm_val <- if (cond_mean_col %in% colnames(df)) df[[cond_mean_col]][i] else NA_real_
    edges_val <- if ("n_edges" %in% colnames(df)) df[["n_edges"]][i] else NA_integer_
    sma_triage_record_r(
      p_plus = p_val,
      cond_cv = cv_val,
      cond_mean_contact = cm_val,
      n_edges = edges_val
    )
  })
  df$regime_code <- vapply(res, `[[`, character(1), "regime_code")
  df$severity_rank <- vapply(res, `[[`, integer(1), "severity_rank")
  df$is_conditioning_trap <- vapply(res, `[[`, logical(1), "is_conditioning_trap")
  df$recommended_plot_encoding <- vapply(res, `[[`, character(1), "recommended_plot_encoding")
  df$action_en <- vapply(res, `[[`, character(1), "action_en")
  df$action_zh <- vapply(res, `[[`, character(1), "action_zh")
  df$inferential_guardrail <- vapply(res, `[[`, character(1), "inferential_guardrail")
  df
}

sma_extract_giotto_inputs <- function(gobject, expression_values = "raw", spatial_network_name = NULL, region_column = NULL) {
  # Supports both native S4 Giotto objects (via GiottoClass/Giotto) and lightweight list-based mock Giotto objects
  if (is.list(gobject) && !isS4(gobject)) {
    counts <- as.matrix(gobject$counts)
    cell_ids <- as.character(gobject$cell_ids)
    genes <- as.character(gobject$genes)
    coords <- as.matrix(gobject$coords)
    adjacency <- as.matrix(gobject$adjacency)
    region_ids <- if (!is.null(gobject$region_ids)) as.character(gobject$region_ids) else rep("region_0", length(cell_ids))
    return(list(
      cell_ids = cell_ids, genes = genes, counts = counts,
      coords = coords, adjacency = adjacency, region_ids = region_ids
    ))
  }
  if (!requireNamespace("GiottoClass", quietly = TRUE)) {
    stop("GiottoClass is required to extract inputs from an S4 Giotto object.")
  }
  pdata <- as.data.frame(GiottoClass::pDataDT(gobject))
  cell_ids <- as.character(pdata$cell_ID)
  region_ids <- if (!is.null(region_column) && (region_column %in% colnames(pdata))) {
    as.character(pdata[[region_column]])
  } else {
    rep("region_0", length(cell_ids))
  }
  expr_obj <- GiottoClass::getExpression(gobject, values = expression_values, output = "matrix")
  expr_mat <- as.matrix(expr_obj)
  # Giotto stores expression as genes x cells
  counts <- t(expr_mat[, match(cell_ids, colnames(expr_mat)), drop = FALSE])
  genes <- colnames(counts)
  locs <- as.data.frame(GiottoClass::getSpatialLocations(gobject, output = "data.table"))
  locs <- locs[match(cell_ids, as.character(locs$cell_ID)), , drop = FALSE]
  coords <- as.matrix(locs[, c("sdimx", "sdimy"), drop = FALSE])
  net_dt <- as.data.frame(GiottoClass::getSpatialNetwork(gobject, name = spatial_network_name, output = "networkDT"))
  n <- length(cell_ids)
  adj <- matrix(0.0, nrow = n, ncol = n, dimnames = list(cell_ids, cell_ids))
  u_idx <- match(as.character(net_dt$from), cell_ids)
  v_idx <- match(as.character(net_dt$to), cell_ids)
  valid <- !is.na(u_idx) & !is.na(v_idx) & (u_idx != v_idx)
  adj[cbind(u_idx[valid], v_idx[valid])] <- 1.0
  adj[cbind(v_idx[valid], u_idx[valid])] <- 1.0
  list(
    cell_ids = cell_ids, genes = genes, counts = counts,
    coords = coords, adjacency = adj, region_ids = region_ids
  )
}

sma_audit_giotto_object <- function(
  gobject,
  rule_spec = sma_stromal_myeloid_rule_spec_r(),
  probabilities = NULL,
  pair = c("stromal", "myeloid"),
  p_grid = c(1.0),
  eligible = NULL,
  spatial_network_name = NULL,
  region_column = NULL
) {
  if (!length(p_grid) || any(!is.finite(p_grid)) || any(p_grid < 0 | p_grid > 1))
    stop("p_grid must contain finite retention values in [0,1].")
  if (!is.null(probabilities) && length(p_grid) != 1L)
    stop("A supplied probability matrix is bound to one retention value; use one p_grid value per call.")
  inputs <- sma_extract_giotto_inputs(
    gobject,
    spatial_network_name = spatial_network_name,
    region_column = region_column
  )
  regions <- unique(inputs$region_ids)
  rows <- list()
  idx <- 1L
  for (p_val in as.numeric(p_grid)) {
    probs <- if (!is.null(probabilities)) {
      as.matrix(probabilities)
    } else {
      sma_evaluate_custom_rule_r(
        counts = inputs$counts,
        genes = inputs$genes,
        eligible = eligible,
        p = p_val,
        rule_spec = rule_spec
      )
    }
    for (reg in regions) {
      mask <- inputs$region_ids == reg
      probs_r <- probs[mask, , drop = FALSE]
      adj_r <- inputs$adjacency[mask, mask, drop = FALSE]
      mom <- sma_pair_availability_and_moments_r(
        probabilities = probs_r,
        adjacency = adj_r,
        col_a = pair[1L],
        col_b = pair[2L]
      )
      rows[[idx]] <- data.frame(
        region_id = reg,
        retention_p = p_val,
        label_a = pair[1L],
        label_b = pair[2L],
        n_cells = sum(mask),
        n_edges = mom$n_edges,
        P_plus = mom$P_plus,
        P_00 = mom$P_00,
        P_pos_0 = mom$P_pos_0,
        P_0_pos = mom$P_0_pos,
        mean_C = mom$mean_C,
        cond_mean_C = mom$cond_mean_C,
        cond_cv_C = mom$cond_cv_C,
        cond_cv_ratio = mom$cond_cv_ratio,
        ratio_cv_method = mom$ratio_cv_method,
        ratio_cv_is_exact = mom$ratio_cv_is_exact,
        moment_status = mom$status,
        mean_Na = mom$mean_Na,
        mean_Nb = mom$mean_Nb,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  out_df <- do.call(rbind, rows)
  sma_triage_table_r(out_df, p_plus_col = "P_plus", cv_col = "cond_cv_ratio", cond_mean_col = "cond_mean_C")
}
