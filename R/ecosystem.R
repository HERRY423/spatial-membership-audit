# Native Giotto/spatstat bridges. Caller configures library paths. No install.
# Each call is ONE observation region and ONE complete label realization.

sma_align_labels <- function(cell_ids, labels) {
  if (!is.character(cell_ids) || anyNA(cell_ids) || any(!nzchar(cell_ids)) || anyDuplicated(cell_ids))
    stop('unique nonempty cell identifiers required')
  if (is.null(names(labels)) || anyDuplicated(names(labels)) || !setequal(names(labels), cell_ids))
    stop('named labels must cover exactly the fixed cell universe')
  labels <- as.character(labels[match(cell_ids, names(labels))])
  labels[is.na(labels)] <- 'UNKNOWN'
  if (any(!nzchar(labels))) stop('empty labels forbidden')
  labels
}

sma_export_singler <- function(prediction, cell_ids, pruned=TRUE) {
  field <- if (pruned) 'pruned.labels' else 'labels'
  if (is.null(prediction[[field]]) || is.null(rownames(prediction))) stop('missing named SingleR labels')
  labels <- setNames(prediction[[field]], rownames(prediction))
  data.frame(cell_id=cell_ids, label=sma_align_labels(cell_ids, labels),
             method='SingleR', label_field=field, score_semantics='correlation_not_probability')
}

sma_export_insitutype <- function(result, cell_ids) {
  if (is.null(result$clust)) stop('InSituType clust missing')
  data.frame(cell_id=cell_ids, label=sma_align_labels(cell_ids, result$clust),
             method='InSituType', score_semantics='assigned_class_prob_is_not_full_probability_matrix')
}

sma_export_insitutype_posterior <- function(result, cell_ids, expected_classes) {
  # Recover all native rounded log weights, including classes with no MAP call.
  # prob alone is only the chosen-class probability and is insufficient.
  if (!length(cell_ids) || anyNA(cell_ids) || any(!nzchar(cell_ids)) ||
      !length(expected_classes) || anyNA(expected_classes) || any(!nzchar(expected_classes)) ||
      'cell_id' %in% expected_classes) stop('nonempty cell/type IDs and unreserved export columns required')
  if (is.null(result$logliks) || is.null(result$logliks_from_lost_celltypes))
    stop('both retained and lost-class loglik matrices required; prob alone is insufficient')
  a <- as.matrix(result$logliks); b <- as.matrix(result$logliks_from_lost_celltypes)
  if (!identical(rownames(a), rownames(b))) stop('retained/lost cell axes disagree')
  ll <- cbind(a, b)
  if (is.null(rownames(ll)) || anyDuplicated(rownames(ll)) ||
      !setequal(rownames(ll), cell_ids) || anyDuplicated(cell_ids) ||
      anyDuplicated(colnames(ll)) || anyDuplicated(expected_classes) ||
      !setequal(colnames(ll), expected_classes)) stop('complete explicit cell/type universe required')
  ll <- ll[match(cell_ids, rownames(ll)), match(expected_classes, colnames(ll)), drop=FALSE]
  if (anyNA(ll) || any(ll == Inf) || any(apply(ll, 1, max) == -Inf))
    stop('invalid log weights or a row with no supported class')
  if (!requireNamespace('InSituType', quietly=TRUE)) stop('installed native InSituType converter required')
  p <- getFromNamespace('logliks2probs', 'InSituType')(ll)
  if (any(!is.finite(p)) || any(p < 0) || max(abs(rowSums(p)-1)) > 1e-10)
    stop('native conversion did not yield categorical rows')
  data.frame(cell_id=cell_ids, p, check.names=FALSE)
}

sma_giotto <- function(gobject, cell_ids, labels, network_name, n_simulations, seed,
                       region_ids, pairs=NULL, null_model='UNVERIFIED_NATIVE_IMPLEMENTATION') {
  if (length(n_simulations)!=1L || is.na(n_simulations) || n_simulations<2 || n_simulations!=as.integer(n_simulations))
    stop('explicit integer n_simulations >=2 required')
  aligned <- sma_align_labels(cell_ids, labels)
  if (length(unique(sma_align_labels(cell_ids, region_ids)))!=1L || anyNA(region_ids))
    stop('Giotto adapter requires exactly one explicit region per call')
  # Check against actual Giotto cell order; do not trust a supplied vector alone.
  actual <- GiottoClass::pDataDT(gobject)$cell_ID
  if (!identical(as.character(actual), as.character(cell_ids))) stop('Giotto cell order mismatch')
  meta <- data.frame(cell_ID=cell_ids, sma_label=aligned)
  gobject <- Giotto::addCellMetadata(gobject, new_metadata=meta, by_column=TRUE, column_cell_ID='cell_ID')
  raw <- Giotto::cellProximityEnrichment(gobject, spatial_network_name=network_name,
    cluster_column='sma_label', number_of_simulations=as.integer(n_simulations),
    adjust_method='none', set_seed=TRUE, seed_number=seed)
  states <- if(is.null(pairs))NULL else do.call(rbind,lapply(pairs,function(pair) {
    if(length(pair)!=2L || pair[1]==pair[2])stop('distinct type pair required')
    key <- paste(sort(pair),collapse='--')
    row <- raw$enrichm_res[as.character(raw$enrichm_res$unified_int)==key,]
    present <- all(pair %in% aligned)
    data.frame(label_a=pair[1],label_b=pair[2],
      status=if(!present)'ABSENT_TYPE' else if(nrow(row)==0L)'NOT_RETURNED_BY_NATIVE' else
        if(is.finite(row$enrichm[1]))'FINITE' else 'NONFINITE_NATIVE_SCORE',
      enrichm=if(nrow(row))row$enrichm[1] else NA_real_)
  }))
  list(raw=raw, pair_states=states, network_name=network_name, n_simulations=n_simulations, seed=seed,
       null_model=null_model, interpretation='native conditional score; no biological significance decision')
}

sma_kcross <- function(xy_um, cell_ids, labels, pair, radii_um, window,
                        correction=c('translate','none')) {
  correction <- match.arg(correction)
  xy_um <- as.matrix(xy_um)
  labs <- sma_align_labels(cell_ids, labels)
  if (ncol(xy_um)!=2L || nrow(xy_um)!=length(cell_ids) || any(!is.finite(xy_um))) stop('finite N by 2 coordinates required')
  if (length(window)!=4L || any(!is.finite(window)) || window[2]<=window[1] || window[4]<=window[3]) stop('explicit rectangle required')
  if (length(pair)!=2L || pair[1]==pair[2]) stop('distinct type pair required')
  if (any(xy_um[,1]<window[1] | xy_um[,1]>window[2] | xy_um[,2]<window[3] | xy_um[,2]>window[4])) stop('points outside supplied window')
  if (length(radii_um)<2L || any(!is.finite(radii_um)) || any(diff(radii_um)<=0) || radii_um[1]!=0 ||
      max(radii_um)>min(window[2]-window[1], window[4]-window[3])/2 ||
      max(abs(diff(radii_um)-diff(radii_um)[1]))>1e-10) stop('spatstat requires regular radii starting at zero within window inradius')
  if (!all(pair %in% labs)) return(data.frame(radius_um=radii_um, K=NA_real_, L=NA_real_, L_minus_r=NA_real_, status='ABSENT_TYPE'))
  win <- spatstat.geom::owin(xrange=window[1:2], yrange=window[3:4])
  pp <- spatstat.geom::ppp(xy_um[,1], xy_um[,2], window=win, marks=factor(labs))
  raw <- spatstat.explore::Kcross(pp, i=pair[1], j=pair[2], r=radii_um, correction=correction)
  field <- if (correction=='translate') 'trans' else 'un'
  kval <- raw[[field]]
  if (is.null(kval)) stop('unsupported native Kcross return schema')
  data.frame(radius_um=raw$r, K=kval, L=sqrt(kval/pi), L_minus_r=sqrt(kval/pi)-raw$r,
             status=ifelse(is.finite(kval),'FINITE','NONFINITE_NATIVE_K'))
}
