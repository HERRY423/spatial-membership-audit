#!/usr/bin/env Rscript
# Native Linux PANORAMIC 0.99.4 (plevritis-lab/panoramic on system R 4.6.1) runner for Step 5A (Item 3)
# Usage:
#   /usr/bin/Rscript run_panoramic.R <track_dir> <out_dir> <track_name> [B_pan=5]
args <- commandArgs(trailingOnly = TRUE)
tdir <- args[1]
odir <- args[2]
tname <- args[3]
B_pan <- if (length(args) >= 4L) as.integer(args[4]) else 5L

suppressPackageStartupMessages({
  library(panoramic)
  library(SpatialExperiment)
  library(SummarizedExperiment)
  library(BiocParallel)
  library(data.table)
  library(jsonlite)
})

dir.create(odir, recursive = TRUE, showWarnings = FALSE)
N_CORES <- max(1L, parallel::detectCores(logical = TRUE))
BPP <- BiocParallel::MulticoreParam(workers = min(N_CORES, 8L))

CLASSES <- c("Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid")
PAIRS_DF <- do.call(rbind, lapply(1:5, function(i) {
  do.call(rbind, lapply((i + 1):6, function(j) {
    data.frame(ct1 = CLASSES[i], ct2 = CLASSES[j], pair = paste0(CLASSES[i], "--", CLASSES[j]), stringsAsFactors = FALSE)
  }))
}))

cores_df <- fread(file.path(tdir, "cores.csv"), data.table = FALSE)
cores_vec <- as.character(cores_df$core)
lt <- fread(file.path(tdir, "label_codes.tsv"), data.table = FALSE)
label_vec <- lt$label[order(lt$code)]
rates <- fromJSON(file.path(tdir, "track_rates.json"))

# Read cells exported to lightweight TSV by run_competitors_5a.py (avoids needing arrow in system R 4.6.1!)
read_side_tsv <- function(side) {
  df <- fread(file.path(odir, sprintf("_cells_%s.tsv", side)), data.table = FALSE)
  df$core <- as.character(df$core)
  nc <- nrow(df)
  ddir <- file.path(tdir, "draws")
  base_codes <- as.integer(readBin(file.path(ddir, sprintf("baseline_%s_codes.bin", side)), "raw", n = nc))
  btag <- rates$schedules[[side]]$mapping$p2_both$bank_tag
  if (is.null(btag) || btag == "native") {
    p2_mat <- matrix(base_codes, nrow = 20L, ncol = nc, byrow = TRUE)
  } else {
    raw_b <- as.integer(readBin(file.path(ddir, sprintf("run_%s_%s.bin", side, btag)), "raw", n = 20L * nc))
    p2_mat <- matrix(raw_b, nrow = 20L, ncol = nc, byrow = TRUE)
  }
  list(df = df, base_codes = base_codes, p2_mat = p2_mat)
}

build_spe_list <- function(df, codes, prefix) {
  labs <- label_vec[codes + 1L]
  idx_by_core <- split(seq_len(nrow(df)), df$core)
  spes <- list()
  snames <- character(0)
  for (cn in cores_vec) {
    ix <- idx_by_core[[cn]]
    if (is.null(ix) || length(ix) < 20L) next
    sname <- paste0(prefix, "_", cn)
    nc <- length(ix)
    # Minimal 1-row dummy assay + spatialCoords + cell_type colData
    mat <- matrix(1L, nrow = 1L, ncol = nc, dimnames = list("g1", paste0("c", seq_len(nc))))
    xy <- cbind(x = as.numeric(df$x_um[ix]), y = as.numeric(df$y_um[ix]))
    rownames(xy) <- colnames(mat)
    spe <- SpatialExperiment::SpatialExperiment(
      assays = list(counts = mat),
      colData = S4Vectors::DataFrame(cell_type = as.character(labs[ix]), sample_id = rep(sname, nc)),
      spatialCoords = xy
    )
    spes[[sname]] <- spe
    snames <- c(snames, sname)
  }
  list(spes = spes, snames = snames)
}

extract_panoramic_matrix <- function(se_stats, prefix) {
  # Extract per-core spatial statistic matrix (cores x 15 pairs) from panoramic_spatialstats output
  mat_out <- matrix(NA_real_, nrow = length(cores_vec), ncol = nrow(PAIRS_DF),
                    dimnames = list(cores_vec, PAIRS_DF$pair))
  rd <- as.data.frame(SummarizedExperiment::rowData(se_stats))
  anames <- SummarizedExperiment::assayNames(se_stats)
  # Pick primary effect assay (yi in PANORAMIC 0.99.4)
  est_assay <- intersect(c("yi", "stat", "est", "estimate", "lce", "enrichment"), anames)
  a_use <- if (length(est_assay) > 0L) est_assay[1] else anames[1]
  A <- SummarizedExperiment::assay(se_stats, a_use)
  cnames <- colnames(A)
  # Map sample names back to cores
  c_ids <- sub(paste0("^", prefix, "_"), "", cnames)
  c1_col <- intersect(c("ct1", "cell_type_1", "from"), names(rd))[1]
  c2_col <- intersect(c("ct2", "cell_type_2", "to"), names(rd))[1]
  if (is.na(c1_col) || is.na(c2_col)) {
    cat("[warn] rowData cols:", paste(names(rd), collapse=","), "assayNames:", paste(anames, collapse=","), "\n")
    return(mat_out)
  }
  for (pi in seq_len(nrow(PAIRS_DF))) {
    a <- PAIRS_DF$ct1[pi]; b <- PAIRS_DF$ct2[pi]
    r1 <- which((rd[[c1_col]] == a & rd[[c2_col]] == b) | (rd[[c1_col]] == b & rd[[c2_col]] == a))
    if (length(r1) == 0L) next
    vals <- if (length(r1) == 1L) as.numeric(A[r1, ]) else colMeans(A[r1, , drop = FALSE], na.rm = TRUE)
    m_idx <- match(c_ids, cores_vec)
    ok <- !is.na(m_idx)
    mat_out[m_idx[ok], pi] <- vals[ok]
  }
  mat_out
}

run_panoramic_pair <- function(sa_df, ca, sb_df, cb, run_meta = FALSE, arm_lbl = "Naive") {
  la <- build_spe_list(sa_df, ca, "A")
  lb <- build_spe_list(sb_df, cb, "B")
  common_cores <- intersect(sub("^A_", "", la$snames), sub("^B_", "", lb$snames))
  spes_all <- c(la$spes[paste0("A_", common_cores)], lb$spes[paste0("B_", common_cores)])
  design <- data.frame(
    sample = c(paste0("A_", common_cores), paste0("B_", common_cores)),
    patient = c(common_cores, common_cores),
    group = factor(c(rep("PlatA", length(common_cores)), rep("PlatB", length(common_cores))), levels = c("PlatA", "PlatB")),
    stringsAsFactors = FALSE
  )
  pairs_in <- PAIRS_DF[, c("ct1", "ct2")]
  prep <- panoramic::panoramic_prepare(
    spe_list = spes_all, design = design, cell_type = "cell_type",
    min_cells = 5L, window = "convex", BPPARAM = BPP
  )
  se_stats <- panoramic::panoramic_spatialstats(
    prep = prep, pairs = pairs_in, radii_um = 25,
    stat = "local_comp_enrichment", nsim = 100L, correction = "translate",
    seed = 123L, boot = "approx", BPPARAM = BPP, verbose = FALSE
  )
  cat(sprintf("  [panoramic %s] assayNames=%s, dim=%dx%d\n",
              arm_lbl, paste(SummarizedExperiment::assayNames(se_stats), collapse=","),
              nrow(se_stats), ncol(se_stats)), flush = TRUE)
  # Split A and B columns
  is_a <- startsWith(colnames(se_stats), "A_")
  is_b <- startsWith(colnames(se_stats), "B_")
  mat_a <- extract_panoramic_matrix(se_stats[, is_a], "A")
  mat_b <- extract_panoramic_matrix(se_stats[, is_b], "B")

  contrast_df <- NULL
  if (run_meta) {
    contrast_df <- tryCatch({
      se_meta <- panoramic::panoramic_meta_mv(
        se = se_stats, patient_col = "patient", group_col = "group", sample_col = "sample",
        tau_structure = "patient", method = "REML", group_tau2 = "none", BPPARAM = BPP
      )
      cdf <- as.data.frame(panoramic::panoramic_extract_contrast(se_meta, what = "contrast"))
      cdf$arm <- arm_lbl
      cdf
    }, error = function(e) {
      cat(sprintf("  [warn] panoramic_meta_mv (%s) error: %s\n", arm_lbl, e$message), flush = TRUE)
      NULL
    })
  }
  list(mat_a = mat_a, mat_b = mat_b, contrast = contrast_df)
}

save_mat <- function(mat, path) {
  fwrite(data.frame(core = rownames(mat), as.data.frame(mat, check.names = FALSE), check.names = FALSE), path)
}

t0 <- proc.time()[["elapsed"]]
cat(sprintf("\n=== Native PANORAMIC 0.99.4 on %s (%d cores, B_pan=%d, workers=%d) ===\n",
            tname, length(cores_vec), B_pan, min(N_CORES, 8L)), flush = TRUE)
sa <- read_side_tsv("a")
sb <- read_side_tsv("b")

# 1. Naive + native paired REML meta-analysis contrast (Δ_bio = 0)
r_nv <- run_panoramic_pair(sa$df, sa$base_codes, sb$df, sb$base_codes, run_meta = TRUE, arm_lbl = "Naive")
save_mat(r_nv$mat_a, file.path(odir, "panoramic_naive_a.csv"))
save_mat(r_nv$mat_b, file.path(odir, "panoramic_naive_b.csv"))

# 2. Split-half 2s default (B_pan draws + draw 1 native paired REML meta-analysis)
pa_arr <- array(NA_real_, dim = c(B_pan, length(cores_vec), nrow(PAIRS_DF)))
pb_arr <- array(NA_real_, dim = c(B_pan, length(cores_vec), nrow(PAIRS_DF)))
c_p2_b1 <- NULL
for (b in seq_len(B_pan)) {
  rb <- run_panoramic_pair(sa$df, sa$p2_mat[b, ], sb$df, sb$p2_mat[b, ], run_meta = (b == 1L), arm_lbl = sprintf("SplitHalf2s_b%d", b))
  pa_arr[b, , ] <- rb$mat_a
  pb_arr[b, , ] <- rb$mat_b
  if (b == 1L) c_p2_b1 <- rb$contrast
}
save_mat(structure(pa_arr[1, , ], dimnames = list(cores_vec, PAIRS_DF$pair)), file.path(odir, "panoramic_p2_b1_a.csv"))
save_mat(structure(pb_arr[1, , ], dimnames = list(cores_vec, PAIRS_DF$pair)), file.path(odir, "panoramic_p2_b1_b.csv"))
agg <- function(arr) {
  m <- apply(arr, c(2, 3), function(v) if (sum(is.finite(v)) >= max(1L, ceiling(0.6 * dim(arr)[1]))) mean(v[is.finite(v)]) else NA_real_)
  dimnames(m) <- list(cores_vec, PAIRS_DF$pair); m
}
save_mat(agg(pa_arr), file.path(odir, "panoramic_p2_mean_a.csv"))
save_mat(agg(pb_arr), file.path(odir, "panoramic_p2_mean_b.csv"))

c_all <- rbind(r_nv$contrast, c_p2_b1)
if (!is.null(c_all) && nrow(c_all) > 0L) {
  fwrite(c_all, file.path(odir, "panoramic_native_paired_meta.csv"))
}
cat(sprintf("DONE %s PANORAMIC in %.1fs\n", tname, proc.time()[["elapsed"]] - t0), flush = TRUE)
