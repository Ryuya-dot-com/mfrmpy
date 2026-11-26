# Independently check the generated agreement examples against irr 0.85.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, packageVersion("irr")=="0.85")
inputs <- read.csv(args[1])
out <- read.csv(gzfile(args[2]))
out <- out[out$replicate==1L,]
matches <- c(alpha=0L, kappa=0L)
for (block in split(inputs, inputs$condition)) {
    label <- block$condition[1]
    x <- as.matrix(block[c("R1","R2","R3")])
    x <- x[,colSums(!is.na(x))>0,drop=FALSE]
    reference <- out[out$condition==label & out$scope=="panel",]
    used <- rowSums(!is.na(x))>1L
    levels <- sort(unique(as.vector(x[used,,drop=FALSE])))
    stopifnot(sum(used)==reference$pairable_units, sum(!is.na(x[used,,drop=FALSE]))==reference$ratings_used)
    for (metric in c("nominal","ordinal","interval")) {
        actual <- reference[[paste0("alpha_",metric)]]
        if(length(levels)==1L) {
            # A zero chance-disagreement denominator is undefined in mfrmpy.
            stopifnot(is.na(actual))
            next
        }
        # irr 0.85's complete-data branch assumes two ratings per unit.
        # An entirely missing row forces its general multiplicity calculation.
        check <- irr::kripp.alpha(rbind(t(x),NA_real_),method=metric)$value
        stopifnot(isTRUE(all.equal(actual,check,tolerance=1e-12)))
        matches["alpha"] <- matches["alpha"]+1L
    }
    for (pair in combn(colnames(x),2,simplify=FALSE)) {
        both <- na.omit(x[,pair,drop=FALSE])
        ref <- out[out$condition==label & out$scope==paste(pair,collapse="–"),]
        if(!nrow(both)) {
            stopifnot(nrow(ref)==0L)
            next
        }
        levels <- sort(unique(as.vector(both)))
        stopifnot(nrow(both)==ref$compared_pairs)
        for (metric in c("nominal","linear","quadratic")) {
            actual <- ref[[paste0("kappa_",metric)]]
            if(length(levels)==1L) {
                stopifnot(is.na(actual))
                next
            }
            # irr weights the observed ranks. Align to the declared rubric spacing.
            if(metric!="nominal" && any(diff(levels)!=1L))
                next
            weight <- c(nominal="unweighted",linear="equal",quadratic="squared")[[metric]]
            check <- irr::kappa2(both,weight=weight)$value
            stopifnot(isTRUE(all.equal(actual,check,tolerance=1e-12)))
            matches["kappa"] <- matches["kappa"]+1L
        }
    }
}
print(matches)
