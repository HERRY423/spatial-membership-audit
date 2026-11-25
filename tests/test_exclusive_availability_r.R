# Independent categorical-state counterexamples for the historical extra same-cell term.
# Optional first argument selects a source file, including an unchanged historical version.
args <- commandArgs(trailingOnly=TRUE)
source(if (length(args)) args[1] else "R/standard_ecosystem.R", encoding="UTF-8")

enumerated_availability <- function(P) {
  n <- nrow(P)
  labels <- as.matrix(expand.grid(rep(list(0:2), n)))
  mass <- cbind(1-rowSums(P), P)
  sum(vapply(seq_len(nrow(labels)), function(k) {
    z <- labels[k, ]
    if (!any(z == 1L) || !any(z == 2L)) return(0)
    prod(mass[cbind(seq_len(n), z+1L)])
  }, numeric(1)))
}

cases <- list(
  single_mixed_cell=list(P=matrix(c(.5,.5), nrow=1), expected=0),
  only_one_cell_can_be_either_class=list(P=rbind(c(.5,.5),c(0,0)), expected=0),
  two_mixed_cells_with_unassigned_mass=list(P=rbind(c(.2,.3),c(.2,.3)), expected=.12),
  two_half_half_cells=list(P=rbind(c(.5,.5),c(.5,.5)), expected=.5),
  distinct_certain_classes=list(P=rbind(c(1,0),c(0,1)), expected=1),
  second_class_absent=list(P=rbind(c(.5,0),c(.2,0)), expected=0)
)
failures <- character()
for (name in names(cases)) {
  case <- cases[[name]]
  P <- case$P
  W <- matrix(1, nrow(P), nrow(P)); diag(W) <- 0
  exact <- enumerated_availability(P)
  stopifnot(abs(exact-case$expected) < 1e-12)
  result <- sma_pair_availability_and_moments_r(P, W, 1, 2)
  total <- result$P_00 + result$P_pos_0 + result$P_0_pos + result$P_plus
  ok <- is.finite(result$P_plus) && abs(result$P_plus-exact) < 1e-12 &&
    abs(total-1) < 1e-12
  if (case$expected == 0) ok <- ok && is.na(result$cond_cv_ratio)
  cat(sprintf('%s: %s; P_plus=%.17g; exact=%.17g; presence_mass=%.17g\n',
              if (ok) 'PASS' else 'FAIL', name, result$P_plus, exact, total))
  if (!ok) failures <- c(failures, name)
}
if (length(failures)) stop(paste('Exclusive availability regressions:', paste(failures, collapse=', ')))
cat(sprintf('PASS: %d mutually-exclusive availability counterexamples\n', length(cases)))
