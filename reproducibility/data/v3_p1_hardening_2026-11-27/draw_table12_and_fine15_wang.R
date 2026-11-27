# P1-4 Phase 2B + P1-8 Real Empirical 15-Subtype Fine-Grained TME Audit:
#   Part A: Generate B=20 draw banks for Split-half 2s (default) & Split-half 1s
#           across all 5 Table 1 & Table 2 cohorts using solved table12_split_half_rates.json.
#   Part B: Real-data 15-subtype fine-grained TME benchmark on Wang HTMA r1 vs r2 (53 eval cores,
#           105 fine cell-type pairs from hpca$main_types) comparing Naive, Soft (tau=0.05),
#           Depth-only 2s (q0=0.75), Depth-only 1s (q0=1.00), Split-half 2s, and Split-half 1s.

suppressPackageStartupMessages({
  library(SingleR)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

root <- "C:/Reproducution"
out_dir <- file.path(root, "review/v3_p1_hardening_2026-11-27")
draw_dir <- file.path(out_dir, "draws")
dir.create(draw_dir, recursive = TRUE, showWarnings = FALSE)

rates <- fromJSON(file.path(out_dir, "table12_split_half_rates.json"))
pr1_dir <- file.path(root, "review/ncs_g2_pr1_2026-11-09")
p1_data <- file.path(root, "review/top_phase1_2026-11-19/data/wang_htma")
p1_wang <- file.path(root, "review/top_phase1_2026-11-19/x2_wang_htma")
p1_pr1  <- file.path(root, "review/top_phase1_2026-11-19/x2_touchstone_pr1")
x5_dir  <- file.path(root, "review/top_phase2_2026-11-19/x5_autoq")
x3_dir  <- file.path(root, "review/top_phase2_2026-11-19/x3_cosmx_if")
x3_sub  <- file.path(x3_dir, "subsets")

t0_all <- proc.time()[["elapsed"]]

thin_sparse <- function(x, q, seed) {
  set.seed(seed)
  if (q >= 1.0 - 1e-12) return(x)
  z <- x
  z@x <- as.numeric(rbinom(length(x@x), size = as.integer(round(x@x)), prob = q))
  drop0(z)
}

# ==============================================================================
# PART A1: Wang HTMA r2 at q=0.1000 (serves BOTH r1 vs r2 and r2 vs y24 Split-half!)
# ==============================================================================
cat("=== [Part A1] Wang HTMA r2 Split-half bank (q=0.1000, B=20) ===\n", flush = TRUE)
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

q_r2_sh <- rates$Wang_HTMA_r1_vs_r2$split_half_2s$q_b
t1 <- proc.time()[["elapsed"]]
con <- file(file.path(draw_dir, "wang_r2_eval_sh_q0100.bin"), "wb")
for (rep in 1:20) {
  set.seed(2026112710L + rep)
  thin <- matrix(rbinom(length(x_r2), size = x_r2, prob = q_r2_sh), nrow = ng_w, dimnames = list(w_genes, NULL))
  writeBin(predict_w(thin), con)
}
close(con)
cat(sprintf("  wrote wang_r2_eval_sh_q0100.bin (q=%.4f, %d cells) in %.1fs\n", q_r2_sh, ncol(x_r2), proc.time()[["elapsed"]] - t1), flush = TRUE)
rm(w_model); gc()

# ==============================================================================
# PART B: Real-Data 15-Subtype Fine-Grained TME Audit on Wang HTMA r1 vs r2 (105 pairs!)
# ==============================================================================
cat("=== [Part B] Wang HTMA r1 vs r2 — 15 Fine-Grained HPCA TME Subtypes ===\n", flush = TRUE)
env <- new.env(parent = emptyenv())
load(file.path(root, "review/gb_parallel_2026-11-04/references/hpca.rda"), envir = env)
hpca <- env$hpca
ref_mat <- as.matrix(hpca$data)
main_types <- as.character(hpca$main_types)

fine15_types <- c(
  "Epithelial_cells", "Endothelial_cells", "Fibroblasts", "Smooth_muscle_cells",
  "Tissue_stem_cells", "MSC", "Keratinocytes", "Neurons",
  "T_cells", "NK_cell", "B_cell",
  "Macrophage", "Monocyte", "DC", "Neutrophils"
)
fine15_labels <- ifelse(main_types %in% fine15_types, main_types, "OTHER")
ov_w15 <- w_genes[w_genes %in% rownames(ref_mat)]
stopifnot(length(ov_w15) == length(w_genes))
ref_w15 <- ref_mat[ov_w15, , drop = FALSE]
colnames(ref_w15) <- sprintf("HPCA_%04d", seq_len(ncol(ref_w15)))

w15_model <- trainSingleR(
  ref = ref_w15, labels = fine15_labels, test.genes = ov_w15, restrict = ov_w15,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
w15_codes <- c("UNRESOLVED", sort(as.character(w15_model$labels$unique)))
fwrite(data.table(code = 0:(length(w15_codes) - 1L), label = w15_codes),
       file.path(out_dir, "wang_fine15_label_codes.tsv"), sep = "\t")

predict_w15 <- function(counts, keep_scores = FALSE) {
  good <- colSums(counts) > 0L
  lab <- rep("UNRESOLVED", ncol(counts))
  sc_mat <- NULL
  if (keep_scores) {
    sc_mat <- matrix(NA_real_, ncol(counts), length(w15_model$labels$unique),
                     dimnames = list(NULL, sort(as.character(w15_model$labels$unique))))
  }
  if (any(good)) {
    p <- classifySingleR(
      test = counts[, good, drop = FALSE], trained = w15_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    lab[good] <- as.character(p$labels)
    if (keep_scores) sc_mat[good, colnames(p$scores)] <- as.matrix(p$scores)
  }
  code <- as.raw(match(lab, w15_codes) - 1L)
  if (keep_scores) list(code = code, label = lab, scores = sc_mat) else code
}

t1 <- proc.time()[["elapsed"]]
b1_15 <- predict_w15(x_r1, keep_scores = TRUE)
b2_15 <- predict_w15(x_r2, keep_scores = TRUE)
fwrite(data.frame(label = b1_15$label, as.data.frame(b1_15$scores, check.names = FALSE)),
       file.path(draw_dir, "wang_fine15_r1_baseline_scores.tsv.gz"), sep = "\t", na = "NA")
fwrite(data.frame(label = b2_15$label, as.data.frame(b2_15$scores, check.names = FALSE)),
       file.path(draw_dir, "wang_fine15_r2_baseline_scores.tsv.gz"), sep = "\t", na = "NA")
cat("  Fine-15 baseline prevalences (r2):\n")
print(round(100 * sort(table(b2_15$label), decreasing = TRUE) / length(b2_15$label), 2))
cat(sprintf("  Fine-15 baselines done in %.1fs\n", proc.time()[["elapsed"]] - t1), flush = TRUE)

# Probe split-half curves for Fine-15 (5000 cells x 3 reps)
probe_grid <- c(1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
probe_w15 <- function(mat, seed_base) {
  set.seed(seed_base)
  idx <- sort(sample.int(ncol(mat), min(5000L, ncol(mat)), replace = FALSE))
  sub <- mat[, idx, drop = FALSE]
  mask <- colSums(sub) > 0L
  psis <- numeric(length(probe_grid))
  for (qi in seq_along(probe_grid)) {
    qv <- probe_grid[qi]
    dr <- numeric(3L)
    for (r in 1:3) {
      set.seed(seed_base + qi * 100L + r)
      th <- if (qv >= 1.0 - 1e-12) sub else matrix(rbinom(length(sub), size = sub, prob = qv), nrow = ng_w, dimnames = list(w_genes, NULL))
      h1 <- matrix(rbinom(length(th), size = th, prob = 0.5), nrow = ng_w, dimnames = list(w_genes, NULL))
      h2 <- th - h1
      dr[r] <- mean(predict_w15(h1)[mask] != predict_w15(h2)[mask])
    }
    psis[qi] <- mean(dr)
  }
  list(q_vals = probe_grid, psi_vals = psis, unresolved_frac = mean(!mask))
}
cf1 <- probe_w15(x_r1, 2026112751L)
cf2 <- probe_w15(x_r2, 2026112752L)
writeLines(toJSON(list(r1 = cf1, r2 = cf2), auto_unbox = TRUE, pretty = TRUE, digits = 8),
           file.path(out_dir, "wang_fine15_split_half_curves.json"))
cat(sprintf("  Fine-15 split-half psi@0.75: r1=%.4f, r2=%.4f (r2@0.10=%.4f)\n",
            cf1$psi_vals[3], cf2$psi_vals[3], cf2$psi_vals[9]), flush = TRUE)

# Draw B=20 banks for Fine-15: Depth-only 1s (r2 at r_depth), Depth-only 2s (r1 at 0.75, r2 at 0.75*r_depth),
# and Split-half clamped/matched r2 at q=0.10 (if psi_r1 > psi_r2@0.10)
x5_p <- fromJSON(file.path(x5_dir, "autoq_params.json"))
r_w <- x5_p$r1_vs_r2$q_medpos

run_w15_bank <- function(mat, tag, q_rate, seed_base, B = 20L) {
  t_b <- proc.time()[["elapsed"]]
  con <- file(file.path(draw_dir, sprintf("wang_fine15_%s.bin", tag)), "wb")
  for (rep in seq_len(B)) {
    set.seed(seed_base + rep)
    thin <- matrix(rbinom(length(mat), size = mat, prob = q_rate), nrow = ng_w, dimnames = list(w_genes, NULL))
    writeBin(predict_w15(thin), con)
  }
  close(con)
  cat(sprintf("  wrote wang_fine15_%s.bin (q=%.5f) in %.1fs\n", tag, q_rate, proc.time()[["elapsed"]] - t_b), flush = TRUE)
}

run_w15_bank(x_r1, "r1_q075", 0.75, 2026112761L, 20L)
run_w15_bank(x_r2, "r2_depth1s", r_w, 2026112762L, 20L)
run_w15_bank(x_r2, "r2_depth2s", 0.75 * r_w, 2026112763L, 20L)
run_w15_bank(x_r2, "r2_sh_q010", 0.10, 2026112764L, 20L)
rm(x_r1, x_r2, w15_model, b1_15, b2_15); gc()

# ==============================================================================
# PART A2: Touchstone Xenium PR1 STJ at clamped q=0.1000 (diagnostic verification)
# ==============================================================================
cat("=== [Part A2] Touchstone Xenium PR1 STJ clamped Split-half diagnostic (q=0.1000) ===\n", flush = TRUE)
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
x_stj <- x_stj[xen_genes, , drop = FALSE]
t1 <- proc.time()[["elapsed"]]
con <- file(file.path(draw_dir, "xenium_pr1_stj_sh_q0100.bin"), "wb")
for (rep in 1:20) {
  writeBin(predict_xen(thin_sparse(x_stj, 0.10, 2026112720L + rep)), con)
}
close(con)
cat(sprintf("  wrote xenium_pr1_stj_sh_q0100.bin in %.1fs\n", proc.time()[["elapsed"]] - t1), flush = TRUE)
rm(x_stj, xen_model); gc()

# ==============================================================================
# PART A3 & A4: Touchstone CosMx PR1 & CosMx Breast q=0.25 (Split-half 2s & 1s)
# ==============================================================================
broad_map <- c(
  "Epithelial_cells" = "Epithelial", "T_cells" = "T_cells", "B_cell" = "B_cell",
  "Macrophage" = "Myeloid", "DC" = "Myeloid", "Monocyte" = "Myeloid", "Neutrophils" = "Myeloid",
  "Endothelial_cells" = "Endothelial", "Fibroblasts" = "Stroma", "Smooth_muscle_cells" = "Stroma"
)
train_labels <- ifelse(main_types %in% names(broad_map), broad_map[main_types], "OTHER")

# A3. CosMx PR1
cat("=== [Part A3] Touchstone CosMx PR1 Split-half 2s (q=0.3231) & 1s (q=0.4178) ===\n", flush = TRUE)
cosmx_genes <- read.delim(file.path(pr1_dir, "scuttle/GSM8513808/genes.tsv"), stringsAsFactors = FALSE)$gene
ov_cosmx <- cosmx_genes[cosmx_genes %in% rownames(ref_mat)]
ref_c <- ref_mat[ov_cosmx, , drop = FALSE]
colnames(ref_c) <- sprintf("HPCA_%04d", seq_len(ncol(ref_c)))
cosmx_model <- trainSingleR(
  ref = ref_c, labels = train_labels, test.genes = ov_cosmx, restrict = ov_cosmx,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
cosmx_codes <- c("UNRESOLVED", sort(as.character(cosmx_model$labels$unique)))
predict_cosmx <- function(z) {
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

for (cfg in list(
  list(tag = "cosmx_pr1_stj_sh2s", q = rates$Touchstone_CosMx_PR1$split_half_2s$q_a, seed = 2026112730L),
  list(tag = "cosmx_pr1_stj_sh1s", q = rates$Touchstone_CosMx_PR1$split_half_1s$q_a, seed = 2026112735L)
)) {
  t1 <- proc.time()[["elapsed"]]
  con <- file(file.path(draw_dir, paste0(cfg$tag, ".bin")), "wb")
  for (rep in 1:20) writeBin(predict_cosmx(thin_sparse(xc_stj, cfg$q, cfg$seed + rep)), con)
  close(con)
  cat(sprintf("  wrote %s.bin (q=%.4f) in %.1fs\n", cfg$tag, cfg$q, proc.time()[["elapsed"]] - t1), flush = TRUE)
}
rm(xc_stj, cosmx_model); gc()

# A4. CosMx Breast q=0.25 (BR1 & BR2)
cat("=== [Part A4] CosMx Breast q=0.25 Split-half 2s & 1s (BR1 & BR2) ===\n", flush = TRUE)
br_genes <- read.delim(file.path(x3_sub, "GSM8513809/genes.tsv"), stringsAsFactors = FALSE)$gene
ov_br <- br_genes[br_genes %in% rownames(ref_mat)]
ref_b <- ref_mat[ov_br, , drop = FALSE]
colnames(ref_b) <- sprintf("HPCA_%04d", seq_len(ncol(ref_b)))
br_model <- trainSingleR(
  ref = ref_b, labels = train_labels, test.genes = ov_br, restrict = ov_br,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
br_codes <- c("UNRESOLVED", sort(as.character(br_model$labels$unique)))
predict_br <- function(z) {
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
  list(name = "CosMx_Breast_q025_BR1", gsm = "GSM8513809", s2 = 2026112740L, s1 = 2026112742L),
  list(name = "CosMx_Breast_q025_BR2", gsm = "GSM8513811", s2 = 2026112744L, s1 = 2026112746L)
)) {
  x <- as(readMM(file.path(x3_sub, sp$gsm, "counts_27fov.mtx")), "CsparseMatrix")
  rownames(x) <- read.delim(file.path(x3_sub, sp$gsm, "genes.tsv"), stringsAsFactors = FALSE)$gene
  x_ov <- x[ov_br, , drop = FALSE]
  for (cfg in list(
    list(tag = paste0(sp$gsm, "_deep_sh2s"), q = rates[[sp$name]]$split_half_2s$q_a, seed = sp$s2),
    list(tag = paste0(sp$gsm, "_deep_sh1s"), q = rates[[sp$name]]$split_half_1s$q_a, seed = sp$s1)
  )) {
    t1 <- proc.time()[["elapsed"]]
    con <- file(file.path(draw_dir, paste0(cfg$tag, ".bin")), "wb")
    for (rep in 1:20) writeBin(predict_br(thin_sparse(x_ov, cfg$q, cfg$seed + rep)), con)
    close(con)
    cat(sprintf("  wrote %s.bin (q=%.4f) in %.1fs\n", cfg$tag, cfg$q, proc.time()[["elapsed"]] - t1), flush = TRUE)
  }
}

cat(sprintf("\nALL PART A & PART B R DRAWS COMPLETED in %.1fs\n", proc.time()[["elapsed"]] - t0_all), flush = TRUE)
