# Exact conditional fits on the same replay designs used by fixed-SD MML.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L, packageVersion("mfrmr")=="0.2.4")
ratings <- read.csv(args[1])
destination <- args[2]
workers <- as.integer(args[3])
levels <- c(paste0("A-R",1:3),paste0("B-R",1:3))
L <- matrix((levels=="B-R3")-(levels=="A-R3"),1,dimnames=list("R3 annual change",levels))

fit_one <- function(d) {
    stopifnot(nrow(d)==240L, length(unique(d$Recording))==96L,
              !anyDuplicated(d[c("Recording","RaterYear")]), setequal(d$RaterYear,levels))
    result <- tryCatch({
        fit <- mfrmr::fit_mfrm(d,person="Recording",facets="RaterYear",score="Score",
                              rating_min=1,rating_max=5,category_policy="preserve",
                              model="RSM",method="CML",reltol=1e-11)
        tab <- mfrmr::mfrm_facet_intervals(fit,"RaterYear",contrasts=L)$table
        stopifnot(nrow(tab)==1L)
        if (isTRUE(tab$CIEligible)) {
            stopifnot(all(is.finite(c(tab$Estimate,tab$SE,tab$Lower,tab$Upper))), tab$SE>0)
            data.frame(estimate=tab$Estimate,se=tab$SE,lower=tab$Lower,upper=tab$Upper,
                       p=2*pnorm(-abs(tab$Estimate/tab$SE)),reason="")
        } else {
            stopifnot(!is.na(tab$InferenceReview), nzchar(tab$InferenceReview))
            data.frame(estimate=tab$Estimate,se=NA_real_,lower=NA_real_,upper=NA_real_,
                       p=NA_real_,reason=tab$InferenceReview)
        }
    }, error=function(err) data.frame(estimate=NA_real_,se=NA_real_,lower=NA_real_,
                     upper=NA_real_,p=NA_real_,reason=paste("CML error:",conditionMessage(err))))
    cases <- unique(d[c("replicate","design","effect")])
    stopifnot(nrow(cases)==1L)
    cbind(cases,result)
}

key <- function(d) with(d,paste(replicate,design,effect))
if (file.exists(destination)) {
    saved <- read.csv(destination)
    complete <- key(saved)
    stopifnot(!anyDuplicated(complete),all(complete %in% key(ratings)))
    ratings <- ratings[!key(ratings) %in% complete,]
}
cases <- split(ratings,key(ratings))
for (batch in split(seq_along(cases),ceiling(seq_along(cases)/30L))) {
    fits <- parallel::mclapply(cases[batch],fit_one,mc.cores=workers)
    stopifnot(all(vapply(fits,is.data.frame,logical(1))))
    append <- file.exists(destination)
    write.table(do.call(rbind,fits),destination,sep=",",row.names=FALSE,
                col.names=!append,append=append,na="")
    cat(max(batch),"/",length(cases)," data sets\n",sep="")
}
