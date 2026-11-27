# Resume script: generate ONLY the remaining 7 draw banks after checkpoint
# (skipping already-complete wang_r2_eval_sh_q0100.bin and wang_fine15_{r1_q075,r2_depth1s,r2_depth2s}.bin)

suppressPackageStartupMessages({
  library(SingleR)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

root <- "C:/Reproducution"
out_dir <- file.path(root, "review/v3_p1_hardening_2026-11-27")
draw_dir <- file.path(out_dir, "draws")
rates <- fromJSON(file.path(out_dir, "table12_split_half_rates.json"))

pr1_dir <- file.path(root, "review/ncs_g2_pr1_2026-11-09")
p1_data <- file.path(root, "review/top_phase1_2026-11-19/data/wang_htma")
p1_wang <- file.path(root, "review/top_phase1_2026-11-19/x2_wang_htma")
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

env <- new.env(parent = emptyenv())
load(file.path(root, "review/gb_parallel_2026-11-04/references/hpca.rda"), envir = env)
hpca <- env$hpca
ref_mat <- as.matrix(hpca$data)
main_types <- as.character(hpca$main_types)

# ------------------------------------------------------------------------------
# 1. Overwrite incomplete wang_fine15_r2_sh_q010.bin (exact 20 draws)
# ------------------------------------------------------------------------------
cat("=== [1/4] Completing wang_fine15_r2_sh_q010.bin (q=0.10, B=20) ===\n", flush = TRUE)
w_genes <- readLines(file.path(p1_wang, "singler_genes_246.txt"))
w_genes <- w_genes[nzchar(w_genes)]
ng_w <- length(w_genes)

fine15_types <- c(
  "Epithelial_cells", "Endothelial_cells", "Fibroblasts", "Smooth_muscle_cells",
  "Tissue_stem_cells", "MSC", "Keratinocytes", "Neurons",
  "T_cells", "NK_cell", "B_cell",
  "Macrophage", "Monocyte", "DC", "Neutrophils"
)
fine15_labels <- ifelse(main_types %in% fine15_types, main_types, "OTHER")
ref_w15 <- ref_mat[w_genes, , drop = FALSE]
colnames(ref_w15) <- sprintf("HPCA_%04d", seq_len(ncol(ref_w15)))
w15_model <- trainSingleR(
  ref = ref_w15, labels = fine15_labels, test.genes = w_genes, restrict = w_genes,
  genes = "de", de.method = "classic", aggr.ref = FALSE, check.missing = TRUE,
  num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
)
w15_codes_tbl <- fread(file.path(out_dir, "wang_fine15_label_codes.tsv"))
w15_codes <- w15_codes_tbl$label[order(w15_codes_tbl$code)]

predict_w15 <- function(counts) {
  good <- colSums(counts) > 0L
  lab <- rep("UNRESOLVED", ncol(counts))
  if (any(good)) {
    p <- classifySingleR(
      test = counts[, good, drop = FALSE], trained = w15_model,
      quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
      fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
      num.threads = 14L, BPPARAM = BiocParallel::SerialParam()
    )
    lab[good] <- as.character(p$labels)
  }
  as.raw(match(lab, w15_codes) - 1L)
}

bin_r2 <- file.path(p1_data, "r2_eval_246.bin")
nc_r2 <- as.integer(file.info(bin_r2)$size / (4.0 * ng_w))
con_r <- file(bin_r2, "rb")
x_r2 <- matrix(readBin(con_r, integer(), n = ng_w * nc_r2, size = 4L, endian = "little"),
               nrow = ng_w, dimnames = list(w_genes, NULL))
close(con_r)

t1 <- proc.time()[["elapsed"]]
out_f <- file.path(draw_dir, "wang_fine15_r2_sh_q010.bin")
con <- file(out_f, "wb")
for (rep in 1:20) {
  set.seed(2026112764L + rep)
  thin <- matrix(rbinom(length(x_r2), size = x_r2, prob = 0.10), nrow = ng_w, dimnames = list(w_genes, NULL))
  writeBin(predict_w15(thin), con)
}
close(con)
stopifnot(file.info(out_f)$size == 20L * nc_r2)
cat(sprintf("  wrote wang_fine15_r2_sh_q010.bin (%d B) in %.1fs\n", file.info(out_f)$size, proc.time()[["elapsed"]] - t1), flush = TRUE)
rm(x_r2, w15_model); gc()

# ------------------------------------------------------------------------------
# 2. Touchstone Xenium PR1 STJ clamped Split-half diagnostic (q=0.1000)
# ------------------------------------------------------------------------------
cat("=== [2/4] Touchstone Xenium PR1 STJ clamped Split-half diagnostic ===\n", flush = TRUE)
xen_model <- readRDS(file.path(pr1_dir, "models/Xenium/trained.rds"))
xen_genes <- read.delim(file.path(pr1_dir, "models/Xenium/genes.tsv"), stringsAsFactors = FALSE)$gene
xen_targets <- c("BACKGROUND::Epithelial_cells", "BACKGROUND::Endothelial_cells", "Myeloid")
predict_xen <- function(z) {
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
out_x <- file.path(draw_dir, "xenium_pr1_stj_sh_q0100.bin")
con <- file(out_x, "wb")
for (rep in 1:20) writeBin(predict_xen(thin_sparse(x_stj, 0.10, 2026112720L + rep)), con)
close(con)
stopifnot(file.info(out_x)$size == 20L * ncol(x_stj))
cat(sprintf("  wrote xenium_pr1_stj_sh_q0100.bin (%d B) in %.1fs\n", file.info(out_x)$size, proc.time()[["elapsed"]] - t1), flush = TRUE)
rm(x_stj, xen_model); gc()

# ------------------------------------------------------------------------------
# 3. Touchstone CosMx PR1 Split-half 2s (q=0.3231) & 1s (q=0.4178)
# ------------------------------------------------------------------------------
broad_map <- c(
  "Epithelial_cells" = "Epithelial", "T_cells" = "T_cells", "B_cell" = "B_cell",
  "Macrophage" = "Myeloid", "DC" = "Myeloid", "Monocyte" = "Myeloid", "Neutrophils" = "Myeloid",
  "Endothelial_cells" = "Endothelial", "Fibroblasts" = "Stroma", "Smooth_muscle_cells" = "Stroma"
)
train_labels <- ifelse(main_types %in% names(broad_map), broad_map[main_types], "OTHER")

cat("=== [3/4] Touchstone CosMx PR1 Split-half 2s & 1s ===\n", flush = TRUE)
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
  out_c <- file.path(draw_dir, paste0(cfg$tag, ".bin"))
  con <- file(out_c, "wb")
  for (rep in 1:20) writeBin(predict_cosmx(thin_sparse(xc_stj, cfg$q, cfg$seed + rep)), con)
  close(con)
  stopifnot(file.info(out_c)$size == 20L * ncol(xc_stj))
  cat(sprintf("  wrote %s.bin (q=%.4f, %d B) in %.1fs\n", cfg$tag, cfg$q, file.info(out_c)$size, proc.time()[["elapsed"]] - t1), flush = TRUE)
}
rm(xc_stj, cosmx_model); gc()

# ------------------------------------------------------------------------------
# 4. CosMx Breast q=0.25 Split-half 2s & 1s (BR1 & BR2)
# ------------------------------------------------------------------------------
cat("=== [4/4] CosMx Breast q=0.25 Split-half 2s & 1s (BR1 & BR2) ===\n", flush = TRUE)
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
    out_b <- file.path(draw_dir, paste0(cfg$tag, ".bin"))
    con <- file(out_b, "wb")
    for (rep in 1:20) writeBin(predict_br(thin_sparse(x_ov, cfg$q, cfg$seed + rep)), con)
    close(con)
    stopifnot(file.info(out_b)$size == 20L * ncol(x_ov))
    cat(sprintf("  wrote %s.bin (q=%.4f, %d B) in %.1fs\n", cfg$tag, cfg$q, file.info(out_b)$size, proc.time()[["elapsed"]] - t1), flush = TRUE)
  }
}

cat(sprintf("\nALL REMAINING 7 DRAW BANKS COMPLETED & VERIFIED in %.1fs\n", proc.time()[["elapsed"]] - t0_all), flush = TRUE)
