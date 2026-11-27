#!/usr/bin/env Rscript
# Native Linux spicyR (Bioconductor) & CRAWDAD (JEFworks-Lab) runner for Step 5A (Item 3)
# Usage:
#   Rscript run_spicyr_crawdad.R <mode: smoke|run> <track_dir> <out_dir> <track_name> [B_spicy=10] [B_crawdad=5]
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
tdir <- args[2]
odir <- args[3]
tname <- args[4]
B_spicy <- if (length(args) >= 5) as.integer(args[5]) else 10L
B_crawdad <- if (length(args) >= 6) as.integer(args[6]) else 5L

suppressPackageStartupMessages({
  library(spicyR)
  library(crawdad)
  library(arrow)
  library(data.table)
  library(jsonlite)
  library(sf)
})

dir.create(odir, recursive = TRUE, showWarnings = FALSE)
N_CORES <- max(1L, parallel::detectCores(logical = TRUE))
CLASSES <- c("Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid")
PAIRS_DF <- do.call(rbind, lapply(1:5, function(i) {
  do.call(rbind, lapply((i + 1):6, function(j) {
    data.frame(a = CLASSES[i], b = CLASSES[j], pair = paste0(CLASSES[i], "--", CLASSES[j]), stringsAsFactors = FALSE)
  }))
}))

# Read track metadata, label codes, and track_rates.json
cores_df <- fread(file.path(tdir, "cores.csv"), data.table = FALSE)
cores_vec <- as.character(cores_df$core)
lt <- fread(file.path(tdir, "label_codes.tsv"), data.table = FALSE)
label_vec <- lt$label[order(lt$code)]
rates <- fromJSON(file.path(tdir, "track_rates.json"))

read_side <- function(side) {
  df <- as.data.frame(read_parquet(file.path(tdir, sprintf("run_%s_cells.parquet", side)),
                                   col_select = c("core", "x_um", "y_um")))
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
  list(df = df, base_codes = base_codes, p2_mat = p2_mat, btag = btag)
}

run_spicyr_one <- function(df, codes, core_subset = NULL) {
  labs <- label_vec[codes + 1L]
  dat <- data.frame(
    imageID = factor(df$core, levels = if (is.null(core_subset)) cores_vec else core_subset),
    cellID = paste0("c", seq_len(nrow(df))),
    imageCellID = paste0(df$core, "_c", seq_len(nrow(df))),
    cellType = factor(labs),
    x = as.numeric(df$x_um),
    y = as.numeric(df$y_um),
    stringsAsFactors = FALSE
  )
  if (!is.null(core_subset)) {
    dat <- dat[as.character(dat$imageID) %in% core_subset, , drop = FALSE]
  }
  # Ensure all 6 target classes appear as factor levels
  dat$cellType <- factor(as.character(dat$cellType), levels = union(CLASSES, unique(as.character(dat$cellType))))
  pw <- spicyR::getPairwise(
    cells = dat,
    imageID = "imageID",
    cellType = "cellType",
    spatialCoords = c("x", "y"),
    r = 25,
    from = CLASSES,
    to = CLASSES,
    cores = N_CORES,
    window = "convex",
    edgeCorrect = TRUE
  )
  pw_mat <- as.matrix(pw)
  use_cores <- if (is.null(core_subset)) cores_vec else core_subset
  out <- matrix(NA_real_, nrow = length(use_cores), ncol = nrow(PAIRS_DF),
                dimnames = list(use_cores, PAIRS_DF$pair))
  rn <- rownames(pw_mat)
  cn <- colnames(pw_mat)
  for (pi in seq_len(nrow(PAIRS_DF))) {
    k1 <- paste0(PAIRS_DF$a[pi], "__", PAIRS_DF$b[pi])
    k2 <- paste0(PAIRS_DF$b[pi], "__", PAIRS_DF$a[pi])
    v1 <- if (k1 %in% cn) pw_mat[match(use_cores, rn), k1] else rep(NA_real_, length(use_cores))
    v2 <- if (k2 %in% cn) pw_mat[match(use_cores, rn), k2] else rep(NA_real_, length(use_cores))
    both_ok <- is.finite(v1) & is.finite(v2)
    res <- rep(NA_real_, length(use_cores))
    res[both_ok] <- 0.5 * (v1[both_ok] + v2[both_ok])
    only1 <- is.finite(v1) & !is.finite(v2); res[only1] <- v1[only1]
    only2 <- !is.finite(v1) & is.finite(v2); res[only2] <- v2[only2]
    out[, pi] <- res
  }
  out
}

extract_crawdad_core <- function(x_c, y_c, labs_c, seed_val = 42L) {
  out_z100 <- setNames(rep(NA_real_, nrow(PAIRS_DF)), PAIRS_DF$pair)
  out_auc  <- setNames(rep(NA_real_, nrow(PAIRS_DF)), PAIRS_DF$pair)
  # Require at least 2 target classes with >= 3 cells in this core
  tab <- table(labs_c)
  pres <- intersect(CLASSES, names(tab)[tab >= 3L])
  if (length(pres) < 2L || length(x_c) < 30L) {
    return(list(z100 = out_z100, auc = out_auc))
  }
  # Deduplicate exact identical coordinates (tiny jitter 1e-3 um if any duplicates)
  dup <- duplicated(cbind(x_c, y_c))
  if (any(dup)) {
    set.seed(seed_val)
    x_c[dup] <- x_c[dup] + runif(sum(dup), -1e-3, 1e-3)
    y_c[dup] <- y_c[dup] + runif(sum(dup), -1e-3, 1e-3)
  }
  res <- tryCatch({
    cells_sf <- crawdad::toSF(pos = data.frame(x = x_c, y = y_c), cellTypes = factor(labs_c))
    shuf <- crawdad::makeShuffledCells(
      cells = cells_sf,
      scales = c(100, 200),
      perms = 2,
      ncores = 1,
      seed = seed_val,
      verbose = FALSE
    )
    tr <- crawdad::findTrends(
      cells = cells_sf,
      neighDist = 25,
      ncores = 1,
      shuffleList = shuf,
      verbose = FALSE,
      returnMeans = TRUE
    )
    df_m <- crawdad::meltResultsList(tr, withPerms = FALSE)
    # df_m columns typically: reference, neighbor, scale, Z (or mean)
    zcol <- intersect(c("Z", "mean", "zscore", "value"), names(df_m))[1]
    if (is.na(zcol)) return(NULL)
    df_m$Zval <- as.numeric(df_m[[zcol]])
    df_m$scale_num <- as.numeric(as.character(df_m$scale))
    list(df = df_m)
  }, error = function(e) NULL)
  if (is.null(res) || is.null(res$df) || nrow(res$df) == 0L) {
    return(list(z100 = out_z100, auc = out_auc))
  }
  df_m <- res$df
  for (pi in seq_len(nrow(PAIRS_DF))) {
    a <- PAIRS_DF$a[pi]; b <- PAIRS_DF$b[pi]
    sub_ab <- df_m[(df_m$reference == a & df_m$neighbor == b) | (df_m$reference == b & df_m$neighbor == a), , drop = FALSE]
    if (nrow(sub_ab) > 0L) {
      s100 <- sub_ab$Zval[abs(sub_ab$scale_num - 100) < 1e-6 & is.finite(sub_ab$Zval)]
      if (length(s100) > 0L) out_z100[pi] <- mean(s100)
      sall <- sub_ab$Zval[is.finite(sub_ab$Zval)]
      if (length(sall) > 0L) out_auc[pi] <- mean(sall)
    }
  }
  list(z100 = out_z100, auc = out_auc)
}

run_crawdad_one <- function(df, codes, core_subset, seed_base = 100L) {
  labs_all <- label_vec[codes + 1L]
  m_z100 <- matrix(NA_real_, nrow = length(core_subset), ncol = nrow(PAIRS_DF),
                   dimnames = list(core_subset, PAIRS_DF$pair))
  m_auc  <- matrix(NA_real_, nrow = length(core_subset), ncol = nrow(PAIRS_DF),
                   dimnames = list(core_subset, PAIRS_DF$pair))
  idx_by_core <- split(seq_len(nrow(df)), df$core)
  res_list <- parallel::mclapply(seq_along(core_subset), function(ci) {
    cn <- core_subset[ci]
    ix <- idx_by_core[[cn]]
    if (is.null(ix) || length(ix) < 30L) {
      return(list(z100 = setNames(rep(NA_real_, nrow(PAIRS_DF)), PAIRS_DF$pair),
                  auc  = setNames(rep(NA_real_, nrow(PAIRS_DF)), PAIRS_DF$pair)))
    }
    extract_crawdad_core(df$x_um[ix], df$y_um[ix], labs_all[ix], seed_val = seed_base + ci)
  }, mc.cores = N_CORES)
  for (ci in seq_along(core_subset)) {
    m_z100[ci, ] <- res_list[[ci]]$z100
    m_auc[ci, ]  <- res_list[[ci]]$auc
  }
  list(z100 = m_z100, auc = m_auc)
}

save_mat <- function(mat, path) {
  df_out <- data.frame(core = rownames(mat), as.data.frame(mat, check.names = FALSE), check.names = FALSE)
  fwrite(df_out, path)
}

sa <- read_side("a")
sb <- read_side("b")

if (mode == "smoke") {
  sub_cores <- head(cores_vec, 2L)
  t0 <- proc.time()[["elapsed"]]
  sp_a <- run_spicyr_one(sa$df, sa$base_codes, core_subset = sub_cores)
  cat(sprintf("[smoke spicyR 2 cores] %.2fs, finite=%d/%d\n", proc.time()[["elapsed"]] - t0, sum(is.finite(sp_a)), length(sp_a)))
  print(round(sp_a[, 1:6], 3))

  t1 <- proc.time()[["elapsed"]]
  cr_a <- run_crawdad_one(sa$df, sa$base_codes, core_subset = sub_cores, seed_base = 42L)
  cat(sprintf("[smoke CRAWDAD 2 cores] %.2fs, finite_z100=%d/%d, finite_auc=%d/%d\n",
              proc.time()[["elapsed"]] - t1, sum(is.finite(cr_a$z100)), length(cr_a$z100),
              sum(is.finite(cr_a$auc)), length(cr_a$auc)))
  print(round(cr_a$z100[, 1:6], 3))
  quit(save = "no", status = 0L)
}

cat(sprintf("\n=== Native spicyR & CRAWDAD on %s (%d cores, B_spicy=%d, B_crawdad=%d) ===\n",
            tname, length(cores_vec), B_spicy, B_crawdad), flush = TRUE)

# 1. spicyR: Naive + Split-half 2s (B1 and mean of B_spicy draws) + native spicyR::spicy paired test
t0 <- proc.time()[["elapsed"]]
sp_na <- run_spicyr_one(sa$df, sa$base_codes)
sp_nb <- run_spicyr_one(sb$df, sb$base_codes)
save_mat(sp_na, file.path(odir, "spicyr_naive_a.csv"))
save_mat(sp_nb, file.path(odir, "spicyr_naive_b.csv"))
cat(sprintf("  [spicyR Naive] done in %.1fs\n", proc.time()[["elapsed"]] - t0), flush = TRUE)

# Native spicyR::spicy paired mixed-effects test (subject = core, condition = platform) on Naive & SplitHalf B=1
run_spicy_mixed <- function(codes_a, codes_b, arm_lbl) {
  da <- data.frame(imageID = paste0("A_", sa$df$core), subject = sa$df$core, condition = "PlatA",
                   cellID = paste0("a", seq_len(nrow(sa$df))),
                   imageCellID = paste0("A_", sa$df$core, "_c", seq_len(nrow(sa$df))),
                   cellType = label_vec[codes_a + 1L], x = sa$df$x_um, y = sa$df$y_um, stringsAsFactors = FALSE)
  db <- data.frame(imageID = paste0("B_", sb$df$core), subject = sb$df$core, condition = "PlatB",
                   cellID = paste0("b", seq_len(nrow(sb$df))),
                   imageCellID = paste0("B_", sb$df$core, "_c", seq_len(nrow(sb$df))),
                   cellType = label_vec[codes_b + 1L], x = sb$df$x_um, y = sb$df$y_um, stringsAsFactors = FALSE)
  comb <- rbind(da, db)
  comb$imageID <- factor(comb$imageID)
  comb$subject <- factor(comb$subject)
  comb$condition <- factor(comb$condition, levels = c("PlatA", "PlatB"))
  comb$cellType <- factor(comb$cellType, levels = union(CLASSES, unique(comb$cellType)))
  res <- tryCatch({
    fit <- spicyR::spicy(
      cells = comb, condition = "condition", subject = "subject",
      imageID = "imageID", cellType = "cellType", spatialCoords = c("x", "y"),
      r = 25, from = CLASSES, to = CLASSES, cores = N_CORES, window = "convex", edgeCorrect = TRUE, verbose = FALSE
    )
    tp <- spicyR::topPairs(fit, n = 100)
    tp$arm <- arm_lbl
    tp
  }, error = function(e) {
    cat(sprintf("  [warn] spicyR::spicy (%s) error: %s\n", arm_lbl, e$message), flush = TRUE)
    NULL
  })
  res
}

sp_mx_nv <- run_spicy_mixed(sa$base_codes, sb$base_codes, "Naive")
sp_mx_p2 <- run_spicy_mixed(sa$p2_mat[1, ], sb$p2_mat[1, ], "SplitHalf2s_draw1")
mx_all <- rbind(sp_mx_nv, sp_mx_p2)
if (!is.null(mx_all) && nrow(mx_all) > 0L) {
  fwrite(mx_all, file.path(odir, "spicyr_native_paired_test.csv"))
}

t1 <- proc.time()[["elapsed"]]
sp_a_arr <- array(NA_real_, dim = c(B_spicy, length(cores_vec), nrow(PAIRS_DF)))
sp_b_arr <- array(NA_real_, dim = c(B_spicy, length(cores_vec), nrow(PAIRS_DF)))
for (b in seq_len(B_spicy)) {
  sp_a_arr[b, , ] <- run_spicyr_one(sa$df, sa$p2_mat[b, ])
  sp_b_arr[b, , ] <- run_spicyr_one(sb$df, sb$p2_mat[b, ])
}
save_mat(structure(sp_a_arr[1, , ], dimnames = list(cores_vec, PAIRS_DF$pair)), file.path(odir, "spicyr_p2_b1_a.csv"))
save_mat(structure(sp_b_arr[1, , ], dimnames = list(cores_vec, PAIRS_DF$pair)), file.path(odir, "spicyr_p2_b1_b.csv"))
sp_p2_a <- apply(sp_a_arr, c(2, 3), function(v) if (sum(is.finite(v)) >= ceiling(0.8 * B_spicy)) mean(v[is.finite(v)]) else NA_real_)
sp_p2_b <- apply(sp_b_arr, c(2, 3), function(v) if (sum(is.finite(v)) >= ceiling(0.8 * B_spicy)) mean(v[is.finite(v)]) else NA_real_)
dimnames(sp_p2_a) <- list(cores_vec, PAIRS_DF$pair)
dimnames(sp_p2_b) <- list(cores_vec, PAIRS_DF$pair)
save_mat(sp_p2_a, file.path(odir, "spicyr_p2_mean_a.csv"))
save_mat(sp_p2_b, file.path(odir, "spicyr_p2_mean_b.csv"))
cat(sprintf("  [spicyR SplitHalf2s B=%d] done in %.1fs\n", B_spicy, proc.time()[["elapsed"]] - t1), flush = TRUE)

# 2. CRAWDAD (multi-scale 100 & 200 um): run on unused_heldout cores on tTMA1 (all cores on nTMA)
crawdad_cores <- if ("in_historical_r1r2" %in% names(cores_df) && any(cores_df$in_historical_r1r2 == "unused_heldout")) {
  as.character(cores_df$core[cores_df$in_historical_r1r2 == "unused_heldout"])
} else {
  cores_vec
}
cat(sprintf("  [CRAWDAD] running multi-scale (100, 200 um) on %d/%d cores, B=%d\n",
            length(crawdad_cores), length(cores_vec), B_crawdad), flush = TRUE)
t2 <- proc.time()[["elapsed"]]
cr_na <- run_crawdad_one(sa$df, sa$base_codes, crawdad_cores, seed_base = 1000L)
cr_nb <- run_crawdad_one(sb$df, sb$base_codes, crawdad_cores, seed_base = 2000L)
save_mat(cr_na$z100, file.path(odir, "crawdad_z100_naive_a.csv"))
save_mat(cr_nb$z100, file.path(odir, "crawdad_z100_naive_b.csv"))
save_mat(cr_na$auc,  file.path(odir, "crawdad_auc_naive_a.csv"))
save_mat(cr_nb$auc,  file.path(odir, "crawdad_auc_naive_b.csv"))
cat(sprintf("  [CRAWDAD Naive] done in %.1fs\n", proc.time()[["elapsed"]] - t2), flush = TRUE)

t3 <- proc.time()[["elapsed"]]
cz_a <- array(NA_real_, dim = c(B_crawdad, length(crawdad_cores), nrow(PAIRS_DF)))
cz_b <- array(NA_real_, dim = c(B_crawdad, length(crawdad_cores), nrow(PAIRS_DF)))
ca_a <- array(NA_real_, dim = c(B_crawdad, length(crawdad_cores), nrow(PAIRS_DF)))
ca_b <- array(NA_real_, dim = c(B_crawdad, length(crawdad_cores), nrow(PAIRS_DF)))
for (b in seq_len(B_crawdad)) {
  ra <- run_crawdad_one(sa$df, sa$p2_mat[b, ], crawdad_cores, seed_base = 3000L + b * 500L)
  rb <- run_crawdad_one(sb$df, sb$p2_mat[b, ], crawdad_cores, seed_base = 8000L + b * 500L)
  cz_a[b, , ] <- ra$z100; cz_b[b, , ] <- rb$z100
  ca_a[b, , ] <- ra$auc;  ca_b[b, , ] <- rb$auc
}
save_mat(structure(cz_a[1, , ], dimnames = list(crawdad_cores, PAIRS_DF$pair)), file.path(odir, "crawdad_z100_p2_b1_a.csv"))
save_mat(structure(cz_b[1, , ], dimnames = list(crawdad_cores, PAIRS_DF$pair)), file.path(odir, "crawdad_z100_p2_b1_b.csv"))
agg_mean <- function(arr) {
  m <- apply(arr, c(2, 3), function(v) if (sum(is.finite(v)) >= max(1L, ceiling(0.6 * dim(arr)[1]))) mean(v[is.finite(v)]) else NA_real_)
  dimnames(m) <- list(crawdad_cores, PAIRS_DF$pair)
  m
}
save_mat(agg_mean(cz_a), file.path(odir, "crawdad_z100_p2_mean_a.csv"))
save_mat(agg_mean(cz_b), file.path(odir, "crawdad_z100_p2_mean_b.csv"))
save_mat(agg_mean(ca_a), file.path(odir, "crawdad_auc_p2_mean_a.csv"))
save_mat(agg_mean(ca_b), file.path(odir, "crawdad_auc_p2_mean_b.csv"))
cat(sprintf("  [CRAWDAD SplitHalf2s B=%d] done in %.1fs\n", B_crawdad, proc.time()[["elapsed"]] - t3), flush = TRUE)
cat(sprintf("DONE %s spicyR+CRAWDAD in %.1fs total\n", tname, proc.time()[["elapsed"]] - t0), flush = TRUE)
