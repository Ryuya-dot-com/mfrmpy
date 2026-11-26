# R comparators for the M4 prespecified interaction and its 12 cells.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args)==2L, packageVersion("TAM")=="4.3.25",
    packageVersion("mfrmr")=="0.2.4", !file.exists(args[2]))
ratings <- read.csv(args[1])
cells <- expand.grid(Rater=paste0("R",1:3), Item=paste0("I",1:4))
ids <- paste(cells$Rater,cells$Item)
key <- function(d) paste(d$Rater,d$Item)
focal <- match(c("R3 I1","R1 I1"),ids)
rating_basis <- rbind(diag(2),-1)
item_basis <- rbind(diag(3),-1)
interaction <- kronecker(item_basis,rating_basis)
location <- cbind(-item_basis[match(cells$Item,paste0("I",1:4)),],
                 -rating_basis[match(cells$Rater,paste0("R",1:3)),], interaction)
A <- array(0,c(12,4,13),dimnames=list(ids,as.character(0:3),paste0("p",1:13)))
B <- array(rep(0:3,each=12),c(12,4,1),dimnames=list(ids,as.character(0:3),"ability"))
for (k in 1:3) {
    A[,k+1,1:11] <- k*location
    A[,k+1,12:13] <- -rep(colSums(rating_basis[seq_len(k),,drop=FALSE]),each=12)
}
nodes <- seq(-8,8,length.out=161)
weights <- dnorm(nodes)/sum(dnorm(nodes))
L <- rbind(interaction, interaction[focal[1],]-interaction[focal[2],])
labels <- rbind(data.frame(rater=cells$Rater,item=cells$Item,kind="cell"),
                data.frame(rater="",item="",kind="contrast"))

fit_tam <- function(d) {
    persons <- sort(unique(d$Person))
    p <- match(d$Person,persons)
    j <- match(key(d),ids)
    stopifnot(length(persons)==60L,nrow(d)==240L,!anyDuplicated(data.frame(p,j)))
    y <- matrix(NA_integer_,length(persons),12,dimnames=list(persons,ids))
    y[cbind(p,j)] <- d$Score
    fit <- TAM::tam.mml(resp=y,A=A,B=B,variance.fixed=matrix(c(1,1,1),1),beta.fixed=matrix(c(1,1,0),1),
        verbose=FALSE,control=list(nodes=nodes,conv=1e-10,convD=1e-10,convM=1e-10,maxiter=2000))
    x <- fit$xsi$xsi
    counts <- matrix(0,length(persons),48)
    counts[cbind(p,j+12*d$Score)] <- 1
    nll <- function(u) {
        pr <- TAM:::tam_calc_prob(1:12,A,matrix(0,12,4),B,u,matrix(nodes),161L,4L)$rprobs
        -sum(log(exp(counts %*% matrix(log(pr),48,161)) %*% weights))
    }
    stopifnot(length(x)==13L,fit$iter<2000L,abs(nll(x)-fit$deviance/2)<1e-7)
    h <- optimHess(x,nll,control=list(ndeps=rep(2e-4,13)))
    v <- chol2inv(chol(h))
    step <- diag(2e-4,13)
    grad <- apply(step,1,function(u) (nll(x+u)-nll(x-u))/4e-4)
    stopifnot(sqrt(drop(crossprod(grad,v %*% grad))) < .001)
    # The first free interaction coefficient is exactly the R1 × I1 cell.
    stopifnot(all(L[1,] == c(1,0,0,0,0,0)))
    invisible(capture.output(quick <- TAM::tam.se(fit)))
    stopifnot(abs(quick$xsi$se[6]-sqrt(1/h[6,6])) < 1e-5)
    data.frame(estimate=drop(L%*%x[6:11]),se=sqrt(diag(L%*%v[6:11,6:11]%*%t(L))),nll=nll(x),
        native_printed_se=c(fit$xsi$se.xsi[6],rep(NA_real_,12)),
        native_quick_se=c(quick$xsi$se[6],rep(NA_real_,12)))
}

fit_mfrmr <- function(d) {
    fit <- suppressWarnings(mfrmr::fit_mfrm(d,"Person",c("Item","Rater"),"Score",model="RSM",method="MML",
        rating_min=0,rating_max=3,quad_points=31L,reltol=1e-12,facet_interactions="Rater:Item"))
    term <- names(fit$config$interaction_specs)
    stopifnot(length(term)==1L)
    fitted_cells <- fit$interactions$effects
    sig <- paste(fitted_cells$FacetA_Level,fitted_cells$FacetB_Level)
    if (fitted_cells$FacetA[1]=="Item") sig <- paste(fitted_cells$FacetB_Level,fitted_cells$FacetA_Level)
    cell_labels <- paste0(fitted_cells$FacetA,"=",fitted_cells$FacetA_Level," | ",
        fitted_cells$FacetB,"=",fitted_cells$FacetB_Level)
    at <- match(ids,sig)
    contrasts <- diag(12)
    contrasts <- rbind(contrasts,contrasts[focal[1],]-contrasts[focal[2],])
    dimnames(contrasts) <- list(c(ids,"Difference"),cell_labels[at])
    t <- suppressWarnings(mfrmr::mfrm_facet_intervals(fit,interaction=term,contrasts=contrasts))$table
    stopifnot(nrow(t)==13L,all(is.finite(t$SE)),all(t$Target==rownames(contrasts)))
    data.frame(estimate=t$Estimate,se=t$SE,nll=fit$opt$value,native_printed_se=NA_real_,native_quick_se=NA_real_)
}

parts <- split(ratings,list(ratings$replicate,ratings$design,ratings$effect),drop=TRUE)
out <- lapply(parts,function(d) {
    settings <- data.frame(replicate=d$replicate[1],design=d$design[1],effect=d$effect[1])
    rbind(cbind(settings,engine="TAM",labels,fit_tam(d)),cbind(settings,engine="mfrmr",labels,fit_mfrmr(d)))
})
write.csv(do.call(rbind,out),args[2],row.names=FALSE)
