args <- commandArgs(trailingOnly=TRUE)
source(if (length(args)) args[1] else "R/standard_ecosystem.R", encoding="UTF-8")
checks <- 0L
close <- function(x,y,tol=2e-10) {
  stopifnot(isTRUE(all.equal(unname(x),unname(y),tolerance=tol,check.attributes=FALSE)))
  checks <<- checks+1L
}
must_fail <- function(expr) {stopifnot(inherits(tryCatch(force(expr),error=function(e)e),"error"));checks<<-checks+1L}
enumerate <- function(P,W) {
  n <- nrow(P); states <- as.matrix(expand.grid(rep(list(0:2),n)))
  vals <- matrix(0,nrow(states),4)
  for (k in seq_len(nrow(states))) {
    z <- states[k,]; a <- as.numeric(z==1); b <- as.numeric(z==2)
    prob <- cbind(1-rowSums(P),P)
    vals[k,] <- c(prod(prob[cbind(seq_len(n),z+1L)]),as.numeric(t(a)%*%W%*%b),sum(a),sum(b))
  }
  pos <- vals[,3]>0 & vals[,4]>0
  pp <- sum(vals[pos,1]); cm <- cs <- delta <- NA_real_
  mu <- rep(NA_real_,3); sig <- matrix(NA_real_,3,3)
  if (pp>0) {
    v <- vals[pos,2:4,drop=FALSE]; mass <- vals[pos,1]/pp
    mu <- colSums(v*mass); sig <- crossprod(sweep(v,2,mu)*sqrt(mass))
    if (mu[1]>0) {g<-c(1/mu[1],-1/mu[2],-1/mu[3]);delta<-sqrt(max(0,as.numeric(t(g)%*%sig%*%g)))}
  }
  list(pp=pp,mu=mu,sigma=sig,delta=delta,
       meanC=sum(vals[,1]*vals[,2]),varC=sum(vals[,1]*vals[,2]^2)-sum(vals[,1]*vals[,2])^2)
}
set.seed(731)
for (i in 1:12) {
  P <- matrix(runif(8,0,.4),4,2)
  W <- matrix(as.numeric(runif(16)<.6),4,4);W[lower.tri(W)]<-t(W)[lower.tri(W)];diag(W)<-0
  e <- enumerate(P,W); r <- sma_pair_availability_and_moments_r(P,W,1,2)
  close(r$P_plus,e$pp);close(r$mean_C,e$meanC);close(r$var_C,e$varC)
  close(c(r$cond_mean_C,r$cond_mean_Na,r$cond_mean_Nb),e$mu)
  close(r$conditional_covariance,e$sigma);close(r$cond_cv_ratio,e$delta)
}
W <- matrix(c(0,1,1,0),2,2)
r <- sma_pair_availability_and_moments_r(rbind(c(.5,.5),c(0,0)),W,1,2)
close(r$P_plus,0);stopifnot(is.na(r$cond_cv_ratio))
r <- sma_pair_availability_and_moments_r(rbind(c(1e-14,0),c(0,1e-14)),W,1,2)
stopifnot(r$P_plus>0);close(r$P_plus/1e-28,1);close(r$cond_cv_ratio,0)
r <- sma_pair_availability_and_moments_r(rbind(c(1e-300,0),c(0,1e-300)),W,1,2)
stopifnot(is.na(r$P_plus),r$P_plus_underflow,r$status=="NUMERICAL_UNDERFLOW");checks<-checks+1L
stopifnot(sma_triage_record_r(1e-14,.1)$regime_code=="TIER_5A_CONDITIONING_TRAP");checks<-checks+1L
tiny <- rbind(c(1e-300,0),c(0,1e-300),c(1,0))
tiny_graph <- matrix(c(0,1,0,1,0,0,0,0,0),3,3)
r <- sma_pair_availability_and_moments_r(tiny,tiny_graph,1,2)
stopifnot(r$P_plus>0,r$contact_product_underflow,is.na(r$cond_mean_C),
          r$status=="NUMERICAL_UNDERFLOW",is.na(r$cond_cv_ratio));checks<-checks+1L
P <- rbind(c(1,0),c(0,0),c(0,1));W3<-matrix(c(0,1,0,1,0,0,0,0,0),3,3)
r <- sma_pair_availability_and_moments_r(P,W3,1,2)
stopifnot(r$status=="ZERO_NUMERATOR_CV_UNDEFINED",is.na(r$cond_cv_ratio),is.na(r$cond_cv_C));checks<-checks+1L
must_fail(sma_pair_availability_and_moments_r(matrix(c(.6,.6,0,0),2,byrow=TRUE),W,1,2))
must_fail(sma_pair_availability_and_moments_r(matrix(.2,2,2),matrix(c(0,1,0,0),2),1,2))
g <- list(type="gene",gene="G",min_count=1L)
rule <- list(classes=list(list(name="A",positive=list(type="all_of",items=list(g,g)))))
q <- sma_evaluate_custom_rule_r(matrix(c(1,2),ncol=1),"G",p=.5,rule_spec=rule)
close(q[,"A"],c(.5,.75))
rule$classes[[1]]$veto <- g
close(sma_evaluate_custom_rule_r(matrix(1,ncol=1),"G",p=.5,rule_spec=rule)[,"A"],0)
rule <- list(classes=list(list(name="A",positive=g),list(name="B",positive=g)))
must_fail(sma_evaluate_custom_rule_r(matrix(1,ncol=1),"G",p=.1,rule_spec=rule))
joint <- list(type="all_of",items=list(g,list(type="gene_sum",genes=c("G","H"),min_sum=2L)))
q <- sma_evaluate_custom_rule_r(matrix(c(1,1),nrow=1),c("G","H"),p=.5,
                              rule_spec=list(classes=list(list(name="A",positive=joint))))
close(q[,"A"],.25)
must_fail(sma_evaluate_custom_rule_r(matrix(c(1,1),nrow=1),c("G","H"),p=.5,max_states=1,
                                   rule_spec=list(classes=list(list(name="A",positive=joint)))))
cat(sprintf('PASS: %d exact-enumeration and failure-path checks\n',checks))
