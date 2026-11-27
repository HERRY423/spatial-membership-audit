#!/usr/bin/env Rscript
# Prospective Frozen-Protocol SingleR Runner for Wang 2025 nTMA (Item 1)
# Executes ONLY frozen DEFAULT_PROTOCOL_V3 + secondary MITM + fixed reference baselines (NO tuning).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("Usage: run_ntma_singler.R <mode:probe|banks> <track_dir> [hpca_rda]")
mode <- args[1]
tdir <- args[2]
hpca_rda <- if (length(args) >= 3L) args[3] else "/content/v3work/hpca.rda"

suppressPackageStartupMessages({
  library(SingleR)
  library(batchelor)
  library(Matrix)
  library(data.table)
  library(jsonlite)
  library(SummarizedExperiment)
})

n_threads <- max(1L, parallel::detectCores(logical = TRUE))
cat(sprintf("=== nTMA SingleR [%s] %s (threads=%d) ===\n", mode, basename(tdir), n_threads), flush = TRUE)

genes <- readLines(file.path(tdir, "genes.txt"))
genes <- genes[nzchar(genes)]
ng <- length(genes)

read_bin_counts <- function(bin_path) {
  sz <- file.info(bin_path)$size
  nc <- as.integer(round(sz / (4.0 * ng)))
  stopifnot(abs(sz - 4.0 * ng * nc) < 0.5)
  con <- file(bin_path, "rb")
  v <- readBin(con, integer(), n = ng * nc, size = 4L, endian = "little")
  close(con)
  matrix(v, nrow = ng, ncol = nc, dimnames = list(genes, NULL))
}

classify_pos_chunked <- function(mat, idx_pos, model, score_cols, label_codes, chunk_size = 50000L, with_scores = FALSE) {
  n <- ncol(mat)
  lab <- rep("UNRESOLVED", n)
  sc_mat <- if (with_scores) matrix(NA_real_, nrow = n, ncol = length(score_cols), dimnames = list(NULL, score_cols)) else NULL
  m <- length(idx_pos)
  if (m > 0L) {
    starts <- seq(1L, m, by = chunk_size)
    for (s in starts) {
      e <- min(m, s + chunk_size - 1L)
      idx <- idx_pos[s:e]
      sub_m <- as.matrix(mat[, idx, drop = FALSE])
      p <- classifySingleR(
        test = sub_m,
        trained = model,
        quantile = 0.8,
        fine.tune = TRUE,
        tune.thresh = 0.05,
        fine.tune.combined = TRUE,
        prune = FALSE,
        check.missing = TRUE,
        num.threads = n_threads,
        BPPARAM = BiocParallel::SerialParam()
      )
      lab[idx] <- as.character(p$labels)
      if (with_scores) {
        sc_mat[idx, score_cols] <- as.matrix(p$scores)[, score_cols, drop = FALSE]
      }
    }
  }
  code <- match(lab, label_codes) - 1L
  stopifnot(!anyNA(code))
  list(labels = lab, codes = as.raw(code), scores = sc_mat)
}

stratified_subsample <- function(core_cell_counts, target_n = 10000L, seed = 20261125L) {
  total <- sum(core_cell_counts)
  if (total <= target_n) return(seq_len(total))
  set.seed(seed)
  raw_alloc <- target_n * (core_cell_counts / total)
  alloc <- as.integer(floor(raw_alloc))
  rem <- target_n - sum(alloc)
  if (rem > 0L) {
    ord <- order(raw_alloc - alloc, decreasing = TRUE)
    alloc[ord[seq_len(rem)]] <- alloc[ord[seq_len(rem)]] + 1L
  }
  idx_out <- integer(target_n)
  cursor <- 1L
  offset <- 0L
  for (ci in seq_along(core_cell_counts)) {
    k <- alloc[ci]
    nc_c <- core_cell_counts[ci]
    if (k > 0L) {
      idx_out[cursor:(cursor + k - 1L)] <- offset + sort(sample.int(nc_c, k, replace = FALSE))
      cursor <- cursor + k
    }
    offset <- offset + nc_c
  }
  idx_out
}

to_log1p_cp10k_sparse <- function(m_int) {
  m_sp <- as(m_int, "CsparseMatrix")
  lib <- Matrix::colSums(m_sp)
  stopifnot(all(lib > 0))
  m_sp@x <- log1p((m_sp@x / rep(lib, diff(m_sp@p))) * 1e4)
  m_sp
}

model_rds <- file.path(tdir, "trained_singler.rds")
if (mode == "probe") {
  env <- new.env(parent = emptyenv())
  load(hpca_rda, envir = env)
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
  stopifnot(all(genes %in% rownames(ref_mat)))

  if (!file.exists(model_rds)) {
    ref_sub <- ref_mat[genes, , drop = FALSE]
    colnames(ref_sub) <- sprintf("HPCA_%04d", seq_len(ncol(ref_sub)))
    model <- trainSingleR(
      ref = ref_sub,
      labels = train_labels,
      test.genes = genes,
      restrict = genes,
      genes = "de",
      de.method = "classic",
      aggr.ref = FALSE,
      check.missing = TRUE,
      num.threads = n_threads,
      BPPARAM = BiocParallel::SerialParam()
    )
    saveRDS(model, model_rds)
  } else {
    model <- readRDS(model_rds)
  }

  label_codes <- c("UNRESOLVED", sort(as.character(model$labels$unique)))
  score_cols <- sort(as.character(model$labels$unique))
  fwrite(data.frame(code = seq_along(label_codes) - 1L, label = label_codes),
         file.path(tdir, "label_codes.tsv"), sep = "\t")

  cores_tbl <- fread(file.path(tdir, "cores.csv"), data.table = FALSE)
  q_grid <- c(1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
  n_reps <- 3L
  probe_res <- list(n_genes = ng, q_grid = q_grid, n_reps = n_reps)

  for (side in c("a", "b")) {
    t0 <- proc.time()[["elapsed"]]
    mat <- read_bin_counts(file.path(tdir, sprintf("run_%s_counts.bin", side)))
    core_counts <- as.integer(cores_tbl[[sprintf("n_cells_%s", side)]])
    stopifnot(sum(core_counts) == ncol(mat))
    tot_all <- colSums(mat)
    medpos_r <- median(tot_all[tot_all > 0L])

    sub_idx <- stratified_subsample(core_counts, target_n = 10000L, seed = ifelse(side == "a", 202611251L, 202611252L))
    sub_mat <- mat[, sub_idx, drop = FALSE]
    rm(mat); gc()

    classifiable <- colSums(sub_mat) > 0L
    ufrac <- mean(!classifiable)
    cond_psi <- numeric(length(q_grid))
    uncond_psi <- numeric(length(q_grid))

    for (qi in seq_along(q_grid)) {
      qv <- q_grid[qi]
      p1 <- 0.5 * qv
      p2_cond <- min(1.0, max(0.0, p1 / (1.0 - p1)))
      dc <- numeric(n_reps); du <- numeric(n_reps)
      for (rep in seq_len(n_reps)) {
        set.seed(202611400L + ifelse(side == "a", 0L, 5000L) + qi * 100L + rep)
        y1 <- rbinom(length(sub_mat), size = sub_mat, prob = p1)
        y2 <- rbinom(length(sub_mat), size = sub_mat - y1, prob = p2_cond)
        m1 <- matrix(y1, nrow = ng, dimnames = list(genes, NULL))
        m2 <- matrix(y2, nrow = ng, dimnames = list(genes, NULL))
        l1 <- classify_pos_chunked(m1, which(colSums(m1) > 0L), model, score_cols, label_codes)$labels
        l2 <- classify_pos_chunked(m2, which(colSums(m2) > 0L), model, score_cols, label_codes)$labels
        dis <- l1 != l2
        du[rep] <- mean(dis)
        dc[rep] <- mean(dis[classifiable])
      }
      cond_psi[qi] <- mean(dc)
      uncond_psi[qi] <- mean(du)
    }
    probe_res[[side]] <- list(
      n_sub = ncol(sub_mat), n_classifiable = sum(classifiable),
      unresolved_frac = ufrac, medpos_full = medpos_r,
      cond_psi = cond_psi, uncond_psi = uncond_psi
    )
    cat(sprintf("  [probe %s] medpos=%.2f, unres=%.3f, psi(1.0)=%.4f, psi(0.75)=%.4f, psi(0.10)=%.4f (%.1fs)\n",
                side, medpos_r, ufrac, cond_psi[1], cond_psi[3], cond_psi[length(q_grid)], proc.time()[["elapsed"]] - t0), flush = TRUE)
    rm(sub_mat); gc()
  }
  writeLines(toJSON(probe_res, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(tdir, "p2_probe_curves.json"))
} else if (mode == "banks") {
  draw_dir <- file.path(tdir, "draws")
  dir.create(draw_dir, recursive = TRUE, showWarnings = FALSE)
  model <- readRDS(model_rds)
  lab_tbl <- fread(file.path(tdir, "label_codes.tsv"), data.table = FALSE)
  label_codes <- lab_tbl$label[order(lab_tbl$code)]
  score_cols <- sort(as.character(model$labels$unique))
  rates <- fromJSON(file.path(tdir, "track_rates.json"))

  mat_a <- read_bin_counts(file.path(tdir, "run_a_counts.bin"))
  mat_b <- read_bin_counts(file.path(tdir, "run_b_counts.bin"))
  nc_a <- ncol(mat_a); nc_b <- ncol(mat_b)

  # Native baselines
  for (sd in list(list("a", mat_a), list("b", mat_b))) {
    s_tag <- sd[[1]]; ms <- sd[[2]]
    gz <- file.path(draw_dir, sprintf("baseline_%s_scores.tsv.gz", s_tag))
    bn <- file.path(draw_dir, sprintf("baseline_%s_codes.bin", s_tag))
    if (!file.exists(gz) || !file.exists(bn)) {
      t0 <- proc.time()[["elapsed"]]
      res <- classify_pos_chunked(ms, which(colSums(ms) > 0L), model, score_cols, label_codes, with_scores = TRUE)
      writeBin(res$codes, bn)
      fwrite(data.frame(label = res$labels, as.data.frame(res$scores, check.names = FALSE)), gz, sep = "\t", na = "NA")
      cat(sprintf("  [base_%s] %d cells in %.1fs\n", s_tag, ncol(ms), proc.time()[["elapsed"]] - t0), flush = TRUE)
    }
  }

  # B3 MNN rescaleBatches
  b3_a <- file.path(draw_dir, "b3_rescale_a.bin")
  b3_b <- file.path(draw_dir, "b3_rescale_b.bin")
  if (!file.exists(b3_a) || !file.exists(b3_b)) {
    t0 <- proc.time()[["elapsed"]]
    pos_a <- which(colSums(mat_a) > 0L); pos_b <- which(colSums(mat_b) > 0L)
    l1 <- to_log1p_cp10k_sparse(mat_a[, pos_a, drop = FALSE])
    l2 <- to_log1p_cp10k_sparse(mat_b[, pos_b, drop = FALSE])
    rb <- batchelor::rescaleBatches(l1, l2)
    corr_assay <- SummarizedExperiment::assay(rb, "corrected")
    n1 <- length(pos_a); n_tot <- ncol(corr_assay)
    preds <- character(n_tot)
    for (s in seq(1L, n_tot, by = 50000L)) {
      e <- min(n_tot, s + 49999L)
      p <- classifySingleR(test = as.matrix(corr_assay[, s:e, drop = FALSE]), trained = model,
                           quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
                           fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
                           num.threads = n_threads, BPPARAM = BiocParallel::SerialParam())
      preds[s:e] <- as.character(p$labels)
    }
    la <- rep("UNRESOLVED", nc_a); la[pos_a] <- preds[seq_len(n1)]
    lb <- rep("UNRESOLVED", nc_b); lb[pos_b] <- preds[seq(n1 + 1L, n_tot)]
    writeBin(as.raw(match(la, label_codes) - 1L), b3_a)
    writeBin(as.raw(match(lb, label_codes) - 1L), b3_b)
    rm(l1, l2, rb, corr_assay); gc()
    cat(sprintf("  [B3_rescaleBatches] %d + %d pos in %.1fs\n", length(pos_a), length(pos_b), proc.time()[["elapsed"]] - t0), flush = TRUE)
  }
  rm(mat_a, mat_b); gc()

  # Scheduled B=20 thinning banks
  B <- 20L
  realized <- list()
  for (side in c("a", "b")) {
    side_idx <- ifelse(side == "a", 1L, 2L)
    sched_df <- as.data.frame(rates$schedules[[side]]$scheduled)
    if (nrow(sched_df) == 0L) next
    mat_s <- read_bin_counts(file.path(tdir, sprintf("run_%s_counts.bin", side)))
    nc_s <- ncol(mat_s); tot0 <- sum(as.numeric(mat_s))
    for (bi in seq_len(nrow(sched_df))) {
      b_tag <- as.character(sched_df$tag[bi])
      q_val <- as.numeric(sched_df$q[bi])
      out_bin <- file.path(draw_dir, sprintf("run_%s_%s.bin", side, b_tag))
      exp_bytes <- as.double(B) * as.double(nc_s)
      if (file.exists(out_bin) && abs(file.info(out_bin)$size - exp_bytes) < 0.5) {
        cat(sprintf("  [skip] %s\n", basename(out_bin)), flush = TRUE)
        next
      }
      t0 <- proc.time()[["elapsed"]]
      tmp_bin <- paste0(out_bin, ".tmp")
      con <- file(tmp_bin, "wb")
      reals <- numeric(B)
      seed_base <- 70000000L + side_idx * 100000L + bi * 1000L
      for (rep in seq_len(B)) {
        set.seed(seed_base + rep)
        thin <- matrix(rbinom(length(mat_s), size = mat_s, prob = q_val), nrow = ng, ncol = nc_s, dimnames = list(genes, NULL))
        reals[rep] <- sum(as.numeric(thin)) / max(tot0, 1.0)
        res_r <- classify_pos_chunked(thin, which(colSums(thin) > 0L), model, score_cols, label_codes)
        writeBin(res_r$codes, con)
      }
      close(con)
      file.rename(tmp_bin, out_bin)
      dt <- proc.time()[["elapsed"]] - t0
      realized[[sprintf("%s_%s", side, b_tag)]] <- list(
        side = side, tag = b_tag, q_target = q_val, q_realized_mean = mean(reals), q_realized_sd = sd(reals), elapsed_sec = round(dt, 2)
      )
      cat(sprintf("  [bank] %s (q=%.5f, real=%.5f, nc=%d) in %.1fs\n", basename(out_bin), q_val, mean(reals), nc_s, dt), flush = TRUE)
    }
    rm(mat_s); gc()
  }
  writeLines(toJSON(realized, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(draw_dir, "realized_retentions.json"))
}
