# Additional R comparators on the unchanged M2 ratings.
args <- commandArgs(trailingOnly=TRUE)
mode <- if (identical(args[1],"multifacet")) "multifacet" else if (length(args)>=4) args[4] else "comparison"
stopifnot(mode %in% c("comparison","erm","multifacet"),
          as.character(packageVersion("eRm")) == "1.0.10")
if (mode != "erm") stopifnot(as.character(packageVersion("mfrmr")) == "0.2.4")
if (mode == "comparison") stopifnot(as.character(packageVersion("rasch")) == "1.12.1")

L <- matrix(c(-1,0,1), 1, dimnames=list("R3_minus_R1", paste0("R", 1:3)))

fit_mfrmr <- function(data, method, facets=c("Item","Rater"), contrast=L, step_facet=NULL) {
    model <- if (is.null(step_facet)) "RSM" else "PCM"
    settings <- list(data=data, person="Person", facets=facets, score="Score",
                     rating_min=0, rating_max=3, keep_original=TRUE, model=model,
                     method=method, reltol=1e-11)
    if (method == "MML") settings$quad_points <- 31
    if (!is.null(step_facet)) settings$step_facet <- step_facet
    fit <- do.call(mfrmr::fit_mfrm, settings)
    raters <- subset(fit$facets$others, Facet == "Rater")
    values <- setNames(raters$Estimate, raters$Level)
    out <- list(estimate=sum(contrast[1,]*values[colnames(contrast)]))
    tryCatch({
        tab <- mfrmr::mfrm_facet_intervals(fit, "Rater", contrasts=contrast, method="model")$table
        if (nrow(tab)!=1 || !isTRUE(tab$CIEligible[1])) {
            stop(paste(tab$Status, tab$InferenceReview, collapse=" "))
        }
        list(estimate=tab$Estimate, se=tab$SE, lower=tab$Lower, upper=tab$Upper, reason="")
    }, error=function(exc) c(out, list(reason=conditionMessage(exc))))
}

fit_rasch <- function(data) {
    data$Rubric <- "shared"
    fit <- rasch::rasch_mfrm(data, person="Person", item="Rubric", score="Score",
                            facets=c("Item","Rater"))
    raters <- fit$facet_effects$Rater
    values <- setNames(raters$severity, raters$level)
    estimate <- sum(L[1,]*values[colnames(L)])
    out <- list(estimate=estimate, informative_persons=fit$est$cluster_support$informative_rows,
                effective_persons=fit$est$cluster_support$effective)
    if (!isTRUE(fit$est$converged)) return(c(out, list(reason="PCML did not converge")))
    # The public fit keeps the full covariance of virtual thresholds. Within
    # each task and step their R3-minus-R1 difference equals the rater contrast.
    threshold <- fit$est$thr
    if (!all(fit$est$m==3) || sum(!is.na(fit$X))!=nrow(data) || nrow(fit$X)!=length(unique(data$Person))) {
        return(c(out, list(reason="calibration changed score support or rating count")))
    }
    calibrated <- fit$X[cbind(match(data$Person, fit$person$id),
                              match(paste(data$Rubric, data$Item, data$Rater, sep=":"), colnames(fit$X)))]
    stopifnot(all(calibrated==data$Score))
    level <- fit$virtual_map$Rater[threshold$item]
    w <- c(R1=-1,R2=0,R3=1)[level]/12
    stopifnot(sum(level=="R3")==12, sum(level=="R1")==12, abs(sum(w*threshold$tau)-estimate)<1e-8)
    cov <- fit$est$cov_tau
    if (!isTRUE(fit$est$cluster_inference) || !all(is.finite(cov)) ||
        !all(is.finite(raters$se[raters$level %in% c("R1","R3")]))) {
        return(c(out, list(reason="participant sandwich covariance unavailable")))
    }
    variance <- drop(crossprod(w, cov %*% w))
    if (!is.finite(variance) || variance<=0) return(c(out, list(reason="contrast variance unavailable")))
    se <- sqrt(variance)
    c(out, list(se=se, lower=estimate-qnorm(.975)*se, upper=estimate+qnorm(.975)*se, reason=""))
}

fit_erm <- function(data, facets=c("Item","Rater"), contrast=L, step_facet=NULL) {
    # Support rows preserve the declared endpoints. Each has conditional
    # probability one at its corresponding extreme total.
    levels <- lapply(data[facets],function(column) sort(unique(as.character(column))))
    cells <- expand.grid(rev(levels),stringsAsFactors=FALSE)[facets]
    cell_id <- function(rows) do.call(paste, rows[facets])
    ids <- cell_id(cells)
    observed <- ids %in% cell_id(data)
    cells <- cells[observed,,drop=FALSE]
    ids <- ids[observed]
    basis <- lapply(levels,contr.sum)
    n_free <- lengths(levels)-1
    step_owners <- if (is.null(step_facet)) "shared" else levels[[step_facet]]
    step_basis <- contr.sum(3)
    W <- do.call(rbind,lapply(seq_along(ids),function(j) {
        loc <- unlist(lapply(facets,function(f) basis[[f]][match(cells[[f]][j],levels[[f]]),]),
                      use.names=FALSE)
        owner <- if (is.null(step_facet)) "shared" else cells[[step_facet]][j]
        do.call(rbind,lapply(1:3,function(k) {
            step <- as.vector(outer(-colSums(step_basis[seq_len(k),,drop=FALSE]),step_owners==owner))
            c(-k*loc,step)
        }))
    }))
    rownames(W) <- paste(rep(ids,each=3), rep(1:3,length(ids)), sep=".c")
    colnames(W) <- paste0("parameter",seq_len(ncol(W)))
    persons <- sort(unique(data$Person))
    y <- matrix(NA_integer_,length(persons),length(ids),dimnames=list(persons,ids))
    ij <- cbind(match(data$Person,persons),match(cell_id(data),ids))
    stopifnot(!anyNA(ij), !anyDuplicated(as.data.frame(ij)))
    y[ij] <- data$Score
    zeros <- any(apply(y,2,min,na.rm=TRUE)>0)
    maxima <- any(apply(y,2,max,na.rm=TRUE)<3)
    y_fit <- if (zeros) rbind(y, rubric_zero=rep(0,length(ids))) else y
    if (maxima) y_fit <- rbind(y_fit, rubric_max=rep(3,length(ids)))
    estimator <- if (is.null(step_facet)) eRm::LRSM else eRm::LPCM
    fit <- estimator(y_fit,W=.1*W,mpoints=1,groupvec=1,se=TRUE)
    stopifnot(isTRUE(all.equal(unname(fit$X[seq_along(persons),]),unname(y),check.attributes=FALSE)),
              length(fit$etapar)==ncol(W), fit$convergence %in% c(1,2))
    h <- fit$hessian/.1^2
    w <- numeric(ncol(W))
    start <- match("Rater",facets)
    offset <- sum(n_free[seq_len(start-1)])
    w[offset+seq_len(n_free[["Rater"]])] <- contrast[1,levels$Rater] %*% basis$Rater
    estimate <- sum(w*.1*fit$etapar)
    se <- sqrt(drop(crossprod(w,chol2inv(chol(h)) %*% w)))
    list(estimate=estimate, se=se, lower=estimate-qnorm(.975)*se, upper=estimate+qnorm(.975)*se,
         zero_pattern_added=zeros, max_pattern_added=maxima, reason="")
}

estimate <- function(data) {
    calls <- list("mfrmr MML"=function() fit_mfrmr(data,"MML"),
                  "mfrmr CML"=function() fit_mfrmr(data,"CML"),
                  "rasch PCML"=function() fit_rasch(data))
    if (mode=="erm") calls <- list("eRm CML"=function() fit_erm(data))
    else if (data$replicate[1]==1 && data$design[1]!="isolated") calls[["eRm CML pilot"]] <- function() fit_erm(data)
    fields <- c("estimate","se","lower","upper","reason","informative_persons","effective_persons",
                "zero_pattern_added","max_pattern_added")
    do.call(rbind, lapply(names(calls),function(engine) {
        out <- tryCatch(suppressWarnings(calls[[engine]]()),error=function(exc) list(reason=conditionMessage(exc)))
        for (name in setdiff(fields,names(out))) out[[name]] <- NA
        data.frame(replicate=data$replicate[1],design=data$design[1],engine=engine,out[fields],check.names=FALSE)
    }))
}

verify_multifacet <- function(destination) {
    variants <- list(RSM4=list(occasion=FALSE,step_facet=NULL),
                     PCM4=list(occasion=FALSE,step_facet="Criterion"),
                     RSM5=list(occasion=TRUE,step_facet=NULL))
    contrast <- matrix(c(-1,1),1,dimnames=list("R2_minus_R1",c("R1","R2")))
    results <- list()
    for (variant in names(variants)) {
        setting <- variants[[variant]]
        set.seed(20261120 + match(variant,names(variants)))
        levels <- list(Item=c("I1","I2"), Rater=c("R1","R2"), Criterion=c("C1","C2"))
        if (setting$occasion) levels$Occasion <- c("O1","O2")
        persons <- sprintf("P%03d",seq_len(60))
        d <- expand.grid(c(list(Person=persons),levels),stringsAsFactors=FALSE)
        ability <- rnorm(length(persons))[match(d$Person,persons)]
        eta <- ability - ifelse(d$Item=="I1",-.35,.35) - ifelse(d$Rater=="R1",-.45,.45) -
               ifelse(d$Criterion=="C1",-.25,.25)
        if (setting$occasion) eta <- eta - ifelse(d$Occasion=="O1",-.15,.15)
        logits <- outer(eta,0:3) - matrix(rep(cumsum(c(0,-.9,0,.9)),each=nrow(d)),ncol=4)
        if (!is.null(setting$step_facet)) {
            affected <- d$Criterion=="C2"
            logits[affected,] <- outer(eta[affected],0:3) -
                matrix(rep(cumsum(c(0,-.3,-.1,.4)),each=sum(affected)),ncol=4)
        }
        prob <- exp(logits)
        prob <- prob/rowSums(prob)
        u <- runif(nrow(d))
        d$Score <- (u>prob[,1])+(u>rowSums(prob[,1:2]))+(u>rowSums(prob[,1:3]))
        a <- fit_erm(d,names(levels),contrast,setting$step_facet)
        b <- fit_mfrmr(d,"CML",names(levels),contrast,setting$step_facet)
        stopifnot(b$reason=="", abs(a$estimate-b$estimate)<.001,
                  max(abs(c(a$lower-b$lower,a$upper-b$upper)))<.002)
        row <- data.frame(variant=variant, model=substr(variant,1,3), facets=1+length(levels),
                          ratings=nrow(d), persons=length(persons), truth=.9,
                          erm_estimate=a$estimate, mfrmr_estimate=b$estimate,
                          erm_lower=a$lower, erm_upper=a$upper,
                          mfrmr_lower=b$lower, mfrmr_upper=b$upper)
        results[[variant]] <- row
    }
    write.csv(do.call(rbind,results),destination,row.names=FALSE)
}

if (mode=="multifacet") {
    verify_multifacet(args[2])
    quit(save="no")
}
ratings <- read.csv(args[1])
destination <- args[2]
workers <- as.integer(args[3])
key <- function(d) paste(d$replicate,d$design)
if (file.exists(destination)) ratings <- ratings[!key(ratings) %in% key(read.csv(destination)),]
groups <- split(ratings,key(ratings))
for (batch in split(seq_along(groups),ceiling(seq_along(groups)/50))) {
    out <- do.call(rbind,parallel::mclapply(groups[batch],estimate,mc.cores=workers))
    append <- file.exists(destination)
    write.table(out,destination,sep=",",col.names=!append,row.names=FALSE,append=append,na="")
    cat(max(batch),"/",length(groups)," data sets\n",sep="")
}
