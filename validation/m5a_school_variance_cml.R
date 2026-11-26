# Conditional-information intervals for the M5a variance and score-missingness analyses.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L, packageVersion("mfrmr")=="0.2.4")
invisible(loadNamespace("mfrmr"))
rows <- read.csv(args[1])
destination <- args[2]
workers <- as.integer(args[3])
contrast <- matrix(c(-1,0,1),1,dimnames=list("R3_minus_R1",paste0("R",1:3)))
condition <- if ("effect" %in% names(rows)) unique(rows$effect) else "ability_only"
stopifnot(length(condition)==1L, condition %in% c("ability_only","r3_stricter"))
by_missingness <- "scenario" %in% names(rows)
condition_column <- if (by_missingness) "scenario" else "sd_B"
expected_rows <- if (by_missingness) 468L else 480L

fit_dff <- function(data) {
    persons <- length(unique(data$Person))
    stopifnot(nrow(data)==expected_rows, persons<=120L,
              by_missingness || persons==120L,
              !anyDuplicated(data[c("Person","Item","Rater","School")]))
    review <- mfrmr::describe_mfrm_data(data,person="Person",facets=c("Item","Rater","School"),
        score="Score",rating_min=1,rating_max=5,category_policy="preserve",
        comparisons=list(facet="Rater",group="School",groups=c("A","B"),
                         contrasts=contrast,model="RSM",method="CML"))
    retained <- review$overview
    stopifnot(retained$Observations==expected_rows, retained$Persons==persons,
              retained$RatingMin==1L, retained$RatingMax==5L, !retained$ScoreRecoded)
    fit <- do.call(mfrmr::fit_mfrm,c(list(data=data,fit_scope="comparisons",comparison_review=review),
                                  review$comparison_review$fit_arguments))
    tab <- mfrmr::mfrm_facet_intervals(fit)$table
    stopifnot(nrow(tab)==1L, tab$Model=="RSM", tab$Method=="CML",
              tab$StepFacet=="shared", tab$Group1=="A", tab$Group2=="B")
    if (tab$CIEligible) {
        lower <- tab$Lower
        upper <- tab$Upper
        se <- tab$SE
        stopifnot(tab$IdentificationBasis=="conditional_support", tab$ConfidenceLevel==.95,
                  is.finite(tab$Estimate), is.finite(se), se>0,
                  is.finite(lower), is.finite(upper))
        p <- 2*pnorm(-abs(tab$Estimate/se))
        reason <- ""
    } else {
        se <- lower <- upper <- p <- NA_real_
        reason <- tab$InferenceReview
        stopifnot(!is.na(reason), nzchar(reason))
    }
    total <- rowsum(data$Score-1L,data$Person)[,1]
    max_total <- 4L*as.numeric(table(data$Person)[names(total)])
    stopifnot(!anyNA(max_total))
    out <- data.frame(replicate=data$replicate[1],estimate=tab$Estimate,
               se=se,lower=lower,upper=upper,p=p,reason=reason,
               rank=tab$Rank,dimension=tab$Dimension,boundary=tab$BoundaryState,
               numerical_state=tab$NumericalState,gradient_max_abs=tab$GradientMaxAbs,
               informative_persons=sum(total>0L & total<max_total))
    out[[condition_column]] <- data[[condition_column]][1]
    out <- out[c("replicate",condition_column,setdiff(names(out),c("replicate",condition_column)))]
    if (condition!="ability_only") out$effect <- condition
    out
}

key <- function(x) paste(x$replicate,x[[condition_column]])
if (file.exists(destination)) {
    saved <- read.csv(destination)
    saved_condition <- if ("effect" %in% names(saved)) unique(saved$effect) else "ability_only"
    stopifnot(identical(saved_condition,condition))
    completed <- key(saved)
    stopifnot(!anyDuplicated(completed), all(completed %in% key(rows)))
    rows <- rows[!key(rows) %in% completed,]
}
groups <- split(rows,key(rows))
for (batch in split(seq_along(groups),ceiling(seq_along(groups)/25L))) {
    fits <- parallel::mclapply(groups[batch],fit_dff,mc.cores=workers)
    stopifnot(all(vapply(fits,is.data.frame,logical(1))))
    append <- file.exists(destination)
    write.table(do.call(rbind,fits),destination,sep=",",col.names=!append,row.names=FALSE,append=append,na="")
    cat(max(batch),"/",length(groups)," data sets\n",sep="")
}
