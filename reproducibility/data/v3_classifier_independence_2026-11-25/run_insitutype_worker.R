#!/usr/bin/env Rscript
# Native Linux InSituType::insitutypeML worker + HPCA reference exporter for Step 7 (Item 5)
# Modes:
#   export_hpca <hpca_rda> <out_dir>
#   probe       <work_dir> <hpca_rda>
#   banks       <work_dir> <hpca_rda>
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]

suppressPackageStartupMessages({
  library(InSituType)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

load_hpca_6lineage <- function(hpca_rda) {
  env <- new.env(parent = emptyenv())
  load(hpca_rda, envir = env)
  hpca <- env$hpca
  ref_mat <- as.matrix(hpca$data)
  mt <- as.character(hpca$main_types)
  bmap <- c(
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
  labels <- ifelse(mt %in% names(bmap), bmap[mt], "OTHER")
  list(mat = ref_mat, labels = labels)
}

build_insitutype_profiles <- function(ref_mat, labels, genes) {
  sub <- ref_mat[genes, , drop = FALSE]
  # Convert log-scale HPCA intensities to positive linear-scale mean expression profiles per lineage
  lin <- pmax(2^sub - 1.0, 0.0)
  cls <- sort(unique(labels))
  prof <- matrix(0.0, nrow = length(genes), ncol = length(cls), dimnames = list(genes, cls))
  for (c in cls) {
    idx <- which(labels == c)
    v <- rowMeans(lin[, idx, drop = FALSE])
    # Scale each lineage profile to unit mean across panel so baseline scale is balanced across classes
    prof[, c] <- pmax(v / max(mean(v), 1e-6), 1e-4)
  }
  prof
}

classify_insitutype_chunked <- function(mat_g_by_c, profiles, label_codes, chunk_size = 40000L, with_probs = FALSE) {
  # mat_g_by_c is (n_genes x n_cells) integer count matrix
  n <- ncol(mat_g_by_c)
  cls_cols <- colnames(profiles)
  lab <- rep("UNRESOLVED", n)
  prob_mat <- if (with_probs) matrix(NA_real_, nrow = n, ncol = length(cls_cols), dimnames = list(NULL, cls_cols)) else NULL
  pos <- which(colSums(mat_g_by_c) > 0L)
  m <- length(pos)
  if (m > 0L) {
    for (s in seq(1L, m, by = chunk_size)) {
      e <- min(m, s + chunk_size - 1L)
      idx <- pos[s:e]
      sub_x <- Matrix::t(Matrix::Matrix(mat_g_by_c[, idx, drop = FALSE], sparse = TRUE))  # cells x genes dgCMatrix
      neg_vec <- pmax(0.01, 0.001 * Matrix::rowSums(sub_x))
      res <- InSituType::insitutypeML(
        x = sub_x,
        neg = as.numeric(neg_vec),
        reference_profiles = profiles
      )
      cl_raw <- as.character(res$clust)
      cl_raw[is.na(cl_raw) | !(cl_raw %in% cls_cols)] <- "UNRESOLVED"
      lab[idx] <- cl_raw
      if (with_probs) {
        pm <- matrix(0.0, nrow = length(idx), ncol = length(cls_cols), dimnames = list(NULL, cls_cols))
        used_ll <- FALSE
        if (!is.null(res$logliks)) {
          ll_raw <- as.matrix(res$logliks)
          # Handle transposed (classes x cells) vs (cells x classes)
          if (nrow(ll_raw) != length(idx) && ncol(ll_raw) == length(idx)) {
            ll_raw <- t(ll_raw)
          }
          if (nrow(ll_raw) == length(idx)) {
            ll <- matrix(-Inf, nrow = length(idx), ncol = length(cls_cols), dimnames = list(NULL, cls_cols))
            cn <- colnames(ll_raw)
            if (!is.null(cn) && any(cn %in% cls_cols)) {
              ov <- intersect(cls_cols, cn)
              ll[, ov] <- ll_raw[, ov, drop = FALSE]
              used_ll <- TRUE
            } else if (ncol(ll_raw) == length(cls_cols)) {
              ll[, ] <- ll_raw
              used_ll <- TRUE
            }
            if (used_ll) {
              rmax <- apply(ll, 1, function(r) if (any(is.finite(r))) max(r[is.finite(r)]) else 0.0)
              ex <- exp(ll - rmax)
              ex[!is.finite(ex)] <- 0.0
              rs <- rowSums(ex)
              pm <- ex / pmax(rs, 1e-12)
              pm[rs <= 0, ] <- NA_real_
            }
          }
        }
        if (!used_ll && !is.null(res$prob)) {
          if (is.matrix(res$prob) || is.data.frame(res$prob)) {
            pr <- as.matrix(res$prob)
            if (nrow(pr) != length(idx) && ncol(pr) == length(idx)) pr <- t(pr)
            cn <- colnames(pr)
            if (!is.null(cn) && any(cn %in% cls_cols)) {
              ov <- intersect(cls_cols, cn)
              pm[, ov] <- pr[, ov, drop = FALSE]
            } else if (ncol(pr) == length(cls_cols)) {
              pm[, ] <- pr
            }
          } else if (is.numeric(res$prob) && length(res$prob) == length(idx)) {
            # 1D top-class posterior probability vector: distribute residual (1 - p_top) uniformly
            p_top <- pmin(pmax(as.numeric(res$prob), 0.0), 1.0)
            k_cls <- length(cls_cols)
            for (r_i in seq_along(idx)) {
              c_nm <- cl_raw[r_i]
              if (c_nm %in% cls_cols) {
                pm[r_i, ] <- (1.0 - p_top[r_i]) / max(1L, k_cls - 1L)
                pm[r_i, c_nm] <- p_top[r_i]
              } else {
                pm[r_i, ] <- 1.0 / k_cls
              }
            }
          }
        }
        prob_mat[idx, cls_cols] <- pm
      }
    }
  }
  code <- match(lab, label_codes) - 1L
  stopifnot(!anyNA(code))
  list(labels = lab, codes = as.raw(code), probs = prob_mat)
}

read_bin_counts <- function(bin_path, ng, genes) {
  sz <- file.info(bin_path)$size
  nc <- as.integer(round(sz / (4.0 * ng)))
  stopifnot(abs(sz - 4.0 * ng * nc) < 0.5)
  con <- file(bin_path, "rb")
  v <- readBin(con, integer(), n = ng * nc, size = 4L, endian = "little")
  close(con)
  matrix(v, nrow = ng, ncol = nc, dimnames = list(genes, NULL))
}

if (mode == "export_hpca") {
  hpca_rda <- args[2]; out_dir <- args[3]
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  h <- load_hpca_6lineage(hpca_rda)
  writeLines(rownames(h$mat), file.path(out_dir, "hpca_genes_all.txt"))
  writeLines(h$labels, file.path(out_dir, "hpca_labels_6lin.txt"))
  # Write float32 binary (genes x samples, column-major -> transposed in Python)
  con <- file(file.path(out_dir, "hpca_data_f32.bin"), "wb")
  writeBin(as.numeric(h$mat), con, size = 4L, endian = "little")
  close(con)
  writeLines(toJSON(list(n_genes = nrow(h$mat), n_samples = ncol(h$mat), classes = sort(unique(h$labels))), auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, "hpca_export_meta.json"))
  cat(sprintf("Exported HPCA reference: %d genes x %d samples\n", nrow(h$mat), ncol(h$mat)), flush = TRUE)
} else if (mode == "probe") {
  wdir <- args[2]; hpca_rda <- args[3]
  genes <- readLines(file.path(wdir, "genes.txt")); genes <- genes[nzchar(genes)]; ng <- length(genes)
  h <- load_hpca_6lineage(hpca_rda)
  profiles <- build_insitutype_profiles(h$mat, h$labels, genes)
  label_codes <- c("UNRESOLVED", colnames(profiles))
  fwrite(data.frame(code = seq_along(label_codes) - 1L, label = label_codes), file.path(wdir, "insitutype_label_codes.tsv"), sep = "\t")
  writeLines(toJSON(list(
    classifier = "InSituType::insitutypeML",
    package_version = as.character(packageVersion("InSituType")),
    reference = "HPCA_6lineage_plus_OTHER",
    n_genes = ng,
    neg_rule = "pmax(0.01, 0.001 * cell_total_counts)",
    profile_scaling = "linear 2^log_hpca - 1 normalized to unit mean per lineage, floor 1e-4"
  ), auto_unbox = TRUE, pretty = TRUE), file.path(wdir, "insitutype_model_config.json"))

  q_grid <- c(1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
  out <- list(n_genes = ng, q_grid = q_grid)
  for (side in c("a", "b")) {
    mat <- read_bin_counts(file.path(wdir, sprintf("run_%s_counts.bin", side)), ng, genes)
    nc <- ncol(mat)
    set.seed(ifelse(side == "a", 2026701L, 2026702L))
    sidx <- if (nc > 10000L) sort(sample.int(nc, 10000L)) else seq_len(nc)
    sub_m <- mat[, sidx, drop = FALSE]
    rm(mat); gc()
    cls <- colSums(sub_m) > 0L
    ufrac <- mean(!cls)
    cpsi <- numeric(length(q_grid))
    for (qi in seq_along(q_grid)) {
      qv <- q_grid[qi]; p1 <- 0.5 * qv; p2 <- min(1.0, max(0.0, p1 / (1.0 - p1)))
      dc <- numeric(3L)
      for (rep in 1:3) {
        set.seed(770000L + ifelse(side == "a", 0L, 5000L) + qi * 100L + rep)
        y1 <- rbinom(length(sub_m), size = sub_m, prob = p1)
        y2 <- rbinom(length(sub_m), size = sub_m - y1, prob = p2)
        c1 <- classify_insitutype_chunked(matrix(y1, nrow = ng, dimnames = list(genes, NULL)), profiles, label_codes)$codes
        c2 <- classify_insitutype_chunked(matrix(y2, nrow = ng, dimnames = list(genes, NULL)), profiles, label_codes)$codes
        dc[rep] <- mean((c1 != c2)[cls])
      }
      cpsi[qi] <- mean(dc)
    }
    out[[side]] <- list(unresolved_frac = ufrac, cond_psi = cpsi)
    cat(sprintf("  [InSituType probe %s %s] psi(1.0)=%.4f, psi(0.75)=%.4f, psi(0.10)=%.4f\n",
                basename(wdir), side, cpsi[1], cpsi[3], cpsi[length(q_grid)]), flush = TRUE)
  }
  writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(wdir, "insitutype_probe.json"))
} else if (mode == "banks") {
  wdir <- args[2]; hpca_rda <- args[3]
  genes <- readLines(file.path(wdir, "genes.txt")); genes <- genes[nzchar(genes)]; ng <- length(genes)
  h <- load_hpca_6lineage(hpca_rda)
  profiles <- build_insitutype_profiles(h$mat, h$labels, genes)
  lt <- fread(file.path(wdir, "insitutype_label_codes.tsv"), data.table = FALSE)
  label_codes <- lt$label[order(lt$code)]
  sched <- fromJSON(file.path(wdir, "insitutype_schedule.json"))
  B <- 20L

  for (side in c("a", "b")) {
    mat <- read_bin_counts(file.path(wdir, sprintf("run_%s_counts.bin", side)), ng, genes)
    nc <- ncol(mat)
    # Native baseline + posterior probability matrix for native soft plug-in!
    t0 <- proc.time()[["elapsed"]]
    res0 <- classify_insitutype_chunked(mat, profiles, label_codes, with_probs = TRUE)
    writeBin(res0$codes, file.path(wdir, sprintf("it_base_%s.bin", side)))
    fwrite(data.frame(label = res0$labels, as.data.frame(res0$probs, check.names = FALSE)),
           file.path(wdir, sprintf("it_base_%s_probs.tsv.gz", side)), sep = "\t", na = "NA")
    cat(sprintf("  [InSituType base %s %s] %d cells in %.1fs\n", basename(wdir), side, nc, proc.time()[["elapsed"]] - t0), flush = TRUE)

    items <- sched[[side]]
    if (length(items) > 0L) {
      for (nm in names(items)) {
        qv <- as.numeric(items[[nm]])
        out_p <- file.path(wdir, sprintf("it_bank_%s_%s.bin", side, nm))
        t1 <- proc.time()[["elapsed"]]
        con <- file(out_p, "wb")
        for (rep in seq_len(B)) {
          set.seed(780000L + ifelse(side == "a", 10000L, 20000L) + match(nm, names(items)) * 100L + rep)
          if (qv >= 1.0 - 1e-6) {
            writeBin(res0$codes, con)
          } else {
            thin <- matrix(rbinom(length(mat), size = mat, prob = qv), nrow = ng, ncol = nc, dimnames = list(genes, NULL))
            writeBin(classify_insitutype_chunked(thin, profiles, label_codes)$codes, con)
          }
        }
        close(con)
        cat(sprintf("  [InSituType bank %s %s_%s] q=%.4f in %.1fs\n", basename(wdir), side, nm, qv, proc.time()[["elapsed"]] - t1), flush = TRUE)
      }
    }
    rm(mat); gc()
  }
}
