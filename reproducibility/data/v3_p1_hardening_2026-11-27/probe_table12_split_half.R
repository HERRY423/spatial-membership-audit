# P1-4 Phase 2A: Compute exact frozen-protocol 9-point split-half disagreement curves
# across all 5 Table 1 & Table 2 cohorts missing Split-half 2s (default):
#   1. Wang HTMA r1 vs r2 (53 eval cores)
#   2. Wang 1-yr r2 vs y24 (30 eval cores)
#   3. Touchstone Xenium PR1 (GSM8513829 vs GSM8513837)
#   4. Touchstone CosMx PR1 (GSM8513808 vs GSM8513827)
#   5. CosMx Breast q=0.25 (GSM8513809 & GSM8513811, 54 FOVs pooled / BR1 / BR2)
# Also inspects HPCA fine TME subtypes on Wang r1_eval for Step 3 planning.

suppressPackageStartupMessages({
  library(SingleR)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

root <- "C:/Reproducution"
out_dir <- file.path(root, "review/v3_p1_hardening_2026-11-27")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pr1_dir <- file.path(root, "review/ncs_g2_pr1_2026-11-09")
p1_data <- file.path(root, "review/top_phase1_2026-11-19/data/wang_htma")
p1_wang <- file.path(root, "review/top_phase1_2026-11-19/x2_wang_htma")
p1_pr1  <- file.path(root, "review/top_phase1_2026-11-19/x2_touchstone_pr1")
x3_dir  <- file.path(root, "review/top_phase2_2026-11-19/x3_cosmx_if")
x3_sub  <- file.path(x3_dir, "subsets")
x3_draw <- file.path(x3_dir, "draws")

probe_grid <- c(1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
n_reps <- 3L
max_cells <- 10000L

strat_subsample <- function(n_cells, max_n, seed) {
  if (n_cells <= max_n) return(seq_len(n_cells))
  set.seed(seed)
  sort(sample.int(n_cells, max_n, replace = FALSE))
}

probe_dense <- function(mat, pred_fn, seed_base) {
  idx <- strat_subsample(ncol(mat), max_cells, seed_base)
  sub <- mat[, idx, drop = FALSE]
  tot <- colSums(sub)
  mask <- tot > 0L
  ufrac <- mean(!mask)
  psis <- numeric(length(probe_grid))
  for (qi in seq_along(probe_grid)) {
    qv <- probe_grid[qi]
    dis_r <- numeric(n_reps)
    for (r in seq_len(n_reps)) {
      set.seed(seed_base + qi * 100L + r)
      th <- if (qv >= 1.0 - 1e-12) sub else matrix(rbinom(length(sub), size = sub, prob = qv), nrow = nrow(sub), dimnames = list(rownames(sub), NULL))
      h1 <- matrix(rbinom(length(th), size = th, prob = 0.5), nrow = nrow(th), dimnames = list(rownames(th), NULL))
      h2 <- th - h1
      l1 <- pred_fn(h1)
      l2 <- pred_fn(h2)
      dis_r[r] <- mean(l1[mask] != l2[mask])
    }
    psis[qi] <- mean(dis_r)
  }
  list(q_vals = probe_grid, psi_vals = psis, unresolved_frac = ufrac, n_probed = ncol(sub), n_classifiable = sum(mask))
}

thin_sparse <- function(x, q, seed) {
  set.seed(seed)
  if (q >= 1.0 - 1e-12) return(x)
  z <- x
  z@x <- as.numeric(rbinom(length(x@x), size = as.integer(round(x@x)), prob = q))
  drop0(z)
}

probe_sparse <- function(mat, pred_fn, seed_base) {
  idx <- strat_subsample(ncol(mat), max_cells, seed_base)
  sub <- mat[, idx, drop = FALSE]
  tot <- Matrix::colSums(sub)
  mask <- tot > 0
  ufrac <- mean(!mask)
  psis <- numeric(length(probe_grid))
  for (qi in seq_along(probe_grid)) {
    qv <- probe_grid[qi]
    dis_r <- numeric(n_reps)
    for (r in seq_len(n_reps)) {
      set.seed(seed_base + qi * 100L + r)
      th <- if (qv >= 1.0 - 1e-12) sub else {
        z <- sub
        z@x <- as.numeric(rbinom(length(sub@x), size = as.integer(round(sub@x)), prob = qv))
        drop0(z)
      }
      h1 <- th
      h1@x <- as.numeric(rbinom(length(th@x), size = as.integer(round(th@x)), prob = 0.5))
      h2 <- th
      h2@x <- as.numeric(th@x - h1@x)
      h1 <- drop0(h1); h2 <- drop0(h2)
      l1 <- pred_fn(h1)
      l2 <- pred_fn(h2)
      dis_r[r] <- mean(l1[mask] != l2[mask])
    }
    psis[qi] <- mean(dis_r)
  }
  list(q_vals = probe_grid, psi_vals = psis, unresolved_frac = ufrac, n_probed = ncol(sub), n_classifiable = sum(mask))
}

results <- list()
t0_all <- proc.time()[["elapsed"]]

# ==============================================================================
# 1 & 2. Wang HTMA r1, r2, y24 (246 genes, exact trained_wang_full.rds)
# ==============================================================================
cat("=== [1-2/5] Probing Wang HTMA r1, r2, y24 ===\n", flush = TRUE)
w_genes <- readLines(file.path(p1_wang, "singler_genes_246.txt"))
w_genes <- w_genes[nzchar(w_genes)]
ng_w <- length(w_genes)
w_model <- readRDS(file.path(p1_wang, "trained_wang_full.rds"))
lab_codes <- fread(file.path(p1_wang, "singler_label_codes.tsv"))
all_w_codes <- lab_codes$label[order(lab_codes$code)]

read_bin_counts <- function(bin_path) {
  sz <- file.info(bin_path)$size
  nc <- as.integer(sz / (4.0 * ng_w))
  con <- file(bin_path, "rb")
  v <- readBin(con, integer(), n = ng_w * nc, size = 4L, endian = "little")
  close(con)
  matrix(v, nrow = ng_w, dimnames = list(w_genes, NULL))
}

predict_w <- function(counts) {
  good <- colSums(counts) > 0L
  lab <- rep("UNRESOLVED", ncol(counts))
  if (any(good)) {
    p <- classifySingleR(
      test = counts[, good, drop = FALSE], trained = w_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    lab[good] <- as.character(p$labels)
  }
  as.raw(match(lab, all_w_codes) - 1L)
}

x_r1 <- read_bin_counts(file.path(p1_data, "r1_eval_246.bin"))
x_r2 <- read_bin_counts(file.path(p1_data, "r2_eval_246.bin"))
x_y24 <- read_bin_counts(file.path(p1_data, "y24_eval_246.bin"))

# Baseline verification on subset of 500 cells
sc1_ref <- fread(file.path(p1_wang, "baseline_r1_eval_scores.tsv.gz"), nrows = 500)
stopifnot(identical(predict_w(x_r1[, 1:500, drop = FALSE]), as.raw(match(sc1_ref$label, all_w_codes) - 1L)))
cat("  Verified Wang SingleR 100% identical to frozen baseline.\n", flush = TRUE)

c_r1 <- probe_dense(x_r1, predict_w, 2026112701L)
c_r2 <- probe_dense(x_r2, predict_w, 2026112702L)
c_y24 <- probe_dense(x_y24, predict_w, 2026112703L)
results$Wang_HTMA_r1_vs_r2 <- list(side_a = "r1", side_b = "r2", curve_a = c_r1, curve_b = c_r2)
results$Wang_HTMA_r2_vs_y24 <- list(side_a = "r2", side_b = "y24", curve_a = c_r2, curve_b = c_y24)
cat(sprintf("  Wang curves done (%.1fs): r1 psi@0.75=%.4f, r2 psi@0.75=%.4f, y24 psi@0.75=%.4f\n",
            proc.time()[["elapsed"]] - t0_all, c_r1$psi_vals[3], c_r2$psi_vals[3], c_y24$psi_vals[3]), flush = TRUE)
rm(x_r1, x_r2, x_y24, w_model); gc()

# ==============================================================================
# 3. Touchstone Xenium PR1 (GSM8513829 STJ vs GSM8513837 WCM)
# ==============================================================================
cat("=== [3/5] Probing Touchstone Xenium PR1 ===\n", flush = TRUE)
xen_model <- readRDS(file.path(pr1_dir, "models/Xenium/trained.rds"))
xen_genes <- read.delim(file.path(pr1_dir, "models/Xenium/genes.tsv"), stringsAsFactors = FALSE)$gene
xen_targets <- c("BACKGROUND::Epithelial_cells", "BACKGROUND::Endothelial_cells", "Myeloid")

predict_xen <- function(z) {
  z <- z[xen_genes, , drop = FALSE]
  usable <- Matrix::colSums(z > 0) >= 20L
  label <- rep("UNRESOLVED", ncol(z))
  if (any(usable)) {
    pred <- classifySingleR(
      test = z[, usable, drop = FALSE], trained = xen_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    native <- as.character(pred$labels)
    delta <- as.numeric(pred$delta.next)
    sc <- as.matrix(pred$scores)
    ms <- apply(sc, 1L, max)
    acc <- is.finite(delta) & delta >= 0.05 & is.finite(ms) & ms > 0
    idx_u <- which(usable)
    label[idx_u[acc]] <- native[acc]
  }
  as.raw(match(label, xen_targets, nomatch = 0L))
}

x_stj <- as(readMM(file.path(pr1_dir, "scuttle/GSM8513829/input.mtx")), "CsparseMatrix")
rownames(x_stj) <- read.delim(file.path(pr1_dir, "scuttle/GSM8513829/genes.tsv"), stringsAsFactors = FALSE)$gene
x_wcm <- as(readMM(file.path(pr1_dir, "scuttle/GSM8513837/input.mtx")), "CsparseMatrix")
rownames(x_wcm) <- read.delim(file.path(pr1_dir, "scuttle/GSM8513837/genes.tsv"), stringsAsFactors = FALSE)$gene
x_stj <- x_stj[xen_genes, , drop = FALSE]
x_wcm <- x_wcm[xen_genes, , drop = FALSE]

old_wcm <- fread(file.path(pr1_dir, "native/GSM8513837/baseline_labels.tsv"), nrows = 500, data.table = FALSE)
stopifnot(identical(predict_xen(x_wcm[, 1:500, drop = FALSE]), as.raw(match(as.character(old_wcm$label), xen_targets, nomatch = 0L))))

c_xstj <- probe_sparse(x_stj, predict_xen, 2026112704L)
c_xwcm <- probe_sparse(x_wcm, predict_xen, 2026112705L)
results$Touchstone_Xenium_PR1 <- list(side_a = "GSM8513829_STJ", side_b = "GSM8513837_WCM", curve_a = c_xstj, curve_b = c_xwcm)
cat(sprintf("  Xenium PR1 curves done (%.1fs): STJ psi@0.75=%.4f, WCM psi@0.75=%.4f\n",
            proc.time()[["elapsed"]] - t0_all, c_xstj$psi_vals[3], c_xwcm$psi_vals[3]), flush = TRUE)
rm(x_stj, x_wcm, xen_model); gc()

# ==============================================================================
# 4 & 5. CosMx PR1 (GSM8513808 vs GSM8513827) + CosMx Breast q=0.25 (BR1 & BR2)
# ==============================================================================
cat("=== [4-5/5] Probing Touchstone CosMx PR1 & CosMx Breast q=0.25 ===\n", flush = TRUE)
env <- new.env(parent = emptyenv())
load(file.path(root, "review/gb_parallel_2026-11-04/references/hpca.rda"), envir = env)
hpca <- env$hpca
ref_mat <- as.matrix(hpca$data)
main_types <- as.character(hpca$main_types)
broad_map <- c(
  "Epithelial_cells" = "Epithelial",
  "T_cells" = "T_cells",
  "B_cell" = "B_cell",
  "Macrophage" = "Myeloid",
  "DC" = "Myeloid",
  "Monocyte" = "Myeloid",
  "Neutrophils" = "Myeloid",
  "Endothelial_cells" = "Endothelial",
  "Fibroblasts" = "Stroma",
  "Smooth_muscle_cells" = "Stroma"
)
train_labels <- ifelse(main_types %in% names(broad_map), broad_map[main_types], "OTHER")

# 4. CosMx PR1
cosmx_genes <- read.delim(file.path(pr1_dir, "scuttle/GSM8513808/genes.tsv"), stringsAsFactors = FALSE)$gene
ov_cosmx <- cosmx_genes[cosmx_genes %in% rownames(ref_mat)]
ref_c <- ref_mat[ov_cosmx, , drop = FALSE]
colnames(ref_c) <- sprintf("HPCA_%04d", seq_len(ncol(ref_c)))
cosmx_model <- trainSingleR(
  ref = ref_c, labels = train_labels, test.genes = ov_cosmx, restrict = ov_cosmx,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
cosmx_codes_tbl <- read.delim(file.path(p1_pr1, "cosmx_label_codes.tsv"), stringsAsFactors = FALSE)
cosmx_codes <- c("UNRESOLVED", sort(as.character(cosmx_model$labels$unique)))
stopifnot(identical(cosmx_codes, as.character(cosmx_codes_tbl$label)))

predict_cosmx <- function(z) {
  if (nrow(z) != length(ov_cosmx) || !identical(rownames(z), ov_cosmx)) z <- z[ov_cosmx, , drop = FALSE]
  usable <- Matrix::colSums(z > 0) > 0L
  label <- rep("UNRESOLVED", ncol(z))
  if (any(usable)) {
    pred <- classifySingleR(
      test = z[, usable, drop = FALSE], trained = cosmx_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    label[usable] <- as.character(pred$labels)
  }
  as.raw(match(label, cosmx_codes, nomatch = 0L) - 1L)
}

xc_stj <- as(readMM(file.path(pr1_dir, "scuttle/GSM8513808/input.mtx")), "CsparseMatrix")
rownames(xc_stj) <- read.delim(file.path(pr1_dir, "scuttle/GSM8513808/genes.tsv"), stringsAsFactors = FALSE)$gene
xc_stj <- xc_stj[ov_cosmx, , drop = FALSE]
xc_wcm <- as(readMM(file.path(pr1_dir, "scuttle/GSM8513827/input.mtx")), "CsparseMatrix")
rownames(xc_wcm) <- read.delim(file.path(pr1_dir, "scuttle/GSM8513827/genes.tsv"), stringsAsFactors = FALSE)$gene
xc_wcm <- xc_wcm[ov_cosmx, , drop = FALSE]

old_cstj <- fread(file.path(p1_pr1, "GSM8513808_cosmx_baseline_scores.tsv.gz"), nrows = 500, data.table = FALSE)
stopifnot(identical(predict_cosmx(xc_stj[, 1:500, drop = FALSE]), as.raw(match(old_cstj$label, cosmx_codes) - 1L)))

c_cstj <- probe_sparse(xc_stj, predict_cosmx, 2026112706L)
c_cwcm <- probe_sparse(xc_wcm, predict_cosmx, 2026112707L)
results$Touchstone_CosMx_PR1 <- list(side_a = "GSM8513808_STJ", side_b = "GSM8513827_WCM", curve_a = c_cstj, curve_b = c_cwcm)
cat(sprintf("  CosMx PR1 curves done (%.1fs): STJ psi@0.75=%.4f, WCM psi@0.75=%.4f\n",
            proc.time()[["elapsed"]] - t0_all, c_cstj$psi_vals[3], c_cwcm$psi_vals[3]), flush = TRUE)
rm(xc_stj, xc_wcm, cosmx_model); gc()

# 5. CosMx Breast q=0.25 (GSM8513809 BR1 & GSM8513811 BR2)
br_genes <- read.delim(file.path(x3_sub, "GSM8513809/genes.tsv"), stringsAsFactors = FALSE)$gene
ov_br <- br_genes[br_genes %in% rownames(ref_mat)]
ref_b <- ref_mat[ov_br, , drop = FALSE]
colnames(ref_b) <- sprintf("HPCA_%04d", seq_len(ncol(ref_b)))
br_model <- trainSingleR(
  ref = ref_b, labels = train_labels, test.genes = ov_br, restrict = ov_br,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
br_codes_tbl <- read.delim(file.path(x3_dir, "cosmx_br_label_codes.tsv"), stringsAsFactors = FALSE)
br_codes <- c("UNRESOLVED", sort(as.character(br_model$labels$unique)))
stopifnot(identical(br_codes, as.character(br_codes_tbl$label)))

predict_br <- function(z) {
  if (nrow(z) != length(ov_br) || !identical(rownames(z), ov_br)) z <- z[ov_br, , drop = FALSE]
  usable <- Matrix::colSums(z > 0) > 0L
  label <- rep("UNRESOLVED", ncol(z))
  if (any(usable)) {
    pred <- classifySingleR(
      test = z[, usable, drop = FALSE], trained = br_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    label[usable] <- as.character(pred$labels)
  }
  as.raw(match(label, br_codes, nomatch = 0L) - 1L)
}

for (sp in list(
  list(name = "CosMx_Breast_q025_BR1", gsm = "GSM8513809", seed_stress = 93000000L, pseed = 2026112708L),
  list(name = "CosMx_Breast_q025_BR2", gsm = "GSM8513811", seed_stress = 94000000L, pseed = 2026112710L)
)) {
  gsm <- sp$gsm
  x <- as(readMM(file.path(x3_sub, gsm, "counts_27fov.mtx")), "CsparseMatrix")
  rownames(x) <- read.delim(file.path(x3_sub, gsm, "genes.tsv"), stringsAsFactors = FALSE)$gene
  x_ov <- x[ov_br, , drop = FALSE]
  x_sh <- thin_sparse(x_ov, 0.25, sp$seed_stress + 1L)
  sh_sc <- fread(file.path(x3_draw, sprintf("%s_shallow25_baseline_scores.tsv.gz", gsm)), nrows = 500, data.table = FALSE)
  stopifnot(identical(predict_br(x_sh[, 1:500, drop = FALSE]), as.raw(match(as.character(sh_sc$label), br_codes) - 1L)))
  c_dp <- probe_sparse(x_ov, predict_br, sp$pseed)
  c_sh <- probe_sparse(x_sh, predict_br, sp$pseed + 1L)
  results[[sp$name]] <- list(side_a = paste0(gsm, "_deep"), side_b = paste0(gsm, "_shallow25"), curve_a = c_dp, curve_b = c_sh)
  cat(sprintf("  %s curves done (%.1fs): deep psi@0.75=%.4f, shal psi@0.75=%.4f\n",
              sp$name, proc.time()[["elapsed"]] - t0_all, c_dp$psi_vals[3], c_sh$psi_vals[3]), flush = TRUE)
}

writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 8),
           file.path(out_dir, "table12_split_half_curves.json"))
cat(sprintf("\nALL 5 TABLE 1-2 SPLIT-HALF PROBE CURVES SAVED in %.1fs\n", proc.time()[["elapsed"]] - t0_all), flush = TRUE)
