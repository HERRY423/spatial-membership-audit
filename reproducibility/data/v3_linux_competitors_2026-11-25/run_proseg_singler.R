#!/usr/bin/env Rscript
# SingleR worker for Step 5B (Proseg v3.2.0 resegmented CosMx Breast Deep vs Shallow)
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]     # "probe" or "banks"
wdir <- args[2]
hpca_rda <- if (length(args) >= 3L) args[3] else "/content/v3work/hpca.rda"

suppressPackageStartupMessages({
  library(SingleR)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

n_threads <- max(1L, parallel::detectCores(logical = TRUE))
genes <- readLines(file.path(wdir, "genes.txt")); genes <- genes[nzchar(genes)]; ng <- length(genes)

read_bin_counts <- function(bin_path) {
  sz <- file.info(bin_path)$size
  nc <- as.integer(round(sz / (4.0 * ng)))
  stopifnot(abs(sz - 4.0 * ng * nc) < 0.5)
  con <- file(bin_path, "rb")
  v <- readBin(con, integer(), n = ng * nc, size = 4L, endian = "little")
  close(con)
  matrix(v, nrow = ng, ncol = nc, dimnames = list(genes, NULL))
}

classify_pos <- function(mat, model, score_cols, label_codes, with_scores = FALSE) {
  n <- ncol(mat)
  lab <- rep("UNRESOLVED", n)
  sc_mat <- if (with_scores) matrix(NA_real_, nrow = n, ncol = length(score_cols), dimnames = list(NULL, score_cols)) else NULL
  pos <- which(colSums(mat) > 0L)
  if (length(pos) > 0L) {
    for (s in seq(1L, length(pos), by = 40000L)) {
      e <- min(length(pos), s + 39999L)
      idx <- pos[s:e]
      p <- classifySingleR(test = as.matrix(mat[, idx, drop = FALSE]), trained = model,
                           quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05,
                           fine.tune.combined = TRUE, prune = FALSE, check.missing = TRUE,
                           num.threads = n_threads, BPPARAM = BiocParallel::SerialParam())
      lab[idx] <- as.character(p$labels)
      if (with_scores) sc_mat[idx, score_cols] <- as.matrix(p$scores)[, score_cols, drop = FALSE]
    }
  }
  list(labels = lab, codes = as.raw(match(lab, label_codes) - 1L), scores = sc_mat)
}

model_rds <- file.path(wdir, "trained_singler.rds")
if (mode == "probe") {
  env <- new.env(parent = emptyenv())
  load(hpca_rda, envir = env)
  hpca <- env$hpca
  ref_mat <- as.matrix(hpca$data)
  mt <- as.character(hpca$main_types)
  bmap <- c("Epithelial_cells"="Epithelial","T_cells"="T_cells","B_cell"="B_cell",
            "Macrophage"="Myeloid","DC"="Myeloid","Monocyte"="Myeloid","Neutrophils"="Myeloid",
            "Endothelial_cells"="Endothelial","Fibroblasts"="Stroma","Smooth_muscle_cells"="Stroma")
  tl <- ifelse(mt %in% names(bmap), bmap[mt], "OTHER")
  ref_sub <- ref_mat[genes, , drop = FALSE]
  colnames(ref_sub) <- sprintf("HPCA_%04d", seq_len(ncol(ref_sub)))
  model <- trainSingleR(ref = ref_sub, labels = tl, test.genes = genes, restrict = genes,
                        genes = "de", de.method = "classic", aggr.ref = FALSE,
                        check.missing = TRUE, num.threads = n_threads, BPPARAM = BiocParallel::SerialParam())
  saveRDS(model, model_rds)
  label_codes <- c("UNRESOLVED", sort(as.character(model$labels$unique)))
  score_cols <- sort(as.character(model$labels$unique))
  fwrite(data.frame(code = seq_along(label_codes) - 1L, label = label_codes), file.path(wdir, "label_codes.tsv"), sep = "\t")

  q_grid <- c(1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
  out <- list(n_genes = ng, q_grid = q_grid)
  for (side in c("deep", "shal")) {
    mat <- read_bin_counts(file.path(wdir, sprintf("%s_counts.bin", side)))
    nc <- ncol(mat)
    set.seed(ifelse(side == "deep", 2026501L, 2026502L))
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
        set.seed(550000L + ifelse(side == "deep", 0L, 5000L) + qi * 100L + rep)
        y1 <- rbinom(length(sub_m), size = sub_m, prob = p1)
        y2 <- rbinom(length(sub_m), size = sub_m - y1, prob = p2)
        c1 <- classify_pos(matrix(y1, nrow = ng, dimnames = list(genes, NULL)), model, score_cols, label_codes)$codes
        c2 <- classify_pos(matrix(y2, nrow = ng, dimnames = list(genes, NULL)), model, score_cols, label_codes)$codes
        dc[rep] <- mean((c1 != c2)[cls])
      }
      cpsi[qi] <- mean(dc)
    }
    out[[side]] <- list(unresolved_frac = ufrac, cond_psi = cpsi)
  }
  writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(wdir, "probe.json"))
} else if (mode == "banks") {
  model <- readRDS(model_rds)
  lt <- fread(file.path(wdir, "label_codes.tsv"), data.table = FALSE)
  label_codes <- lt$label[order(lt$code)]
  score_cols <- sort(as.character(model$labels$unique))
  sched <- fromJSON(file.path(wdir, "schedule.json"))
  B <- 20L
  for (side in c("deep", "shal")) {
    mat <- read_bin_counts(file.path(wdir, sprintf("%s_counts.bin", side)))
    nc <- ncol(mat)
    res0 <- classify_pos(mat, model, score_cols, label_codes, with_scores = TRUE)
    writeBin(res0$codes, file.path(wdir, sprintf("base_%s.bin", side)))
    fwrite(data.frame(label = res0$labels, as.data.frame(res0$scores, check.names = FALSE)),
           file.path(wdir, sprintf("base_%s_scores.tsv.gz", side)), sep = "\t", na = "NA")
    items <- sched[[side]]
    if (length(items) > 0L) {
      for (nm in names(items)) {
        qv <- as.numeric(items[[nm]])
        out_p <- file.path(wdir, sprintf("bank_%s_%s.bin", side, nm))
        con <- file(out_p, "wb")
        for (rep in seq_len(B)) {
          set.seed(560000L + ifelse(side == "deep", 10000L, 20000L) + match(nm, names(items)) * 100L + rep)
          if (qv >= 1.0 - 1e-6) {
            writeBin(res0$codes, con)
          } else {
            thin <- matrix(rbinom(length(mat), size = mat, prob = qv), nrow = ng, ncol = nc, dimnames = list(genes, NULL))
            writeBin(classify_pos(thin, model, score_cols, label_codes)$codes, con)
          }
        }
        close(con)
        cat(sprintf("  [proseg bank] %s %s q=%.4f done\n", basename(wdir), basename(out_p), qv), flush = TRUE)
      }
    }
    rm(mat); gc()
  }
}
