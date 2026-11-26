# Match the joint contrast where possible and retain the two mfrmr DFF scales.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, packageVersion("mfrmr")=="0.2.4",packageVersion("TAM")=="4.3.25")
data <- read.csv(args[1])
cells <- expand.grid(Item=paste0("I",1:4),Rater=paste0("R",1:3),School=c("A","B"))
keys <- function(d) with(d,paste(Item,Rater,School))
items <- rbind(diag(3),-1)[match(cells$Item,paste0("I",1:4)),]
raters <- rbind(diag(2),-1)[match(cells$Rater,paste0("R",1:3)),]
school <- cbind(cells$School=="A",cells$School=="B")
interaction <- raters*(school[,1]-school[,2])
locations <- cbind(-items,-raters,school,interaction)
steps <- rbind(diag(3),-1)
A <- array(0,c(24,5,12),dimnames=list(keys(cells),as.character(0:4),paste0("p",1:12)))
B <- array(rep(0:4,each=24),c(24,5,1))
for (k in 1:4) {
    A[,k+1,1:9] <- k*locations
    A[,k+1,10:12] <- -rep(colSums(steps[seq_len(k),,drop=FALSE]),each=24)
}
nodes <- seq(-8,8,length.out=161)
weights <- dnorm(nodes)/sum(dnorm(nodes))
joint_weight <- c(0,0,0,0,0,0,0,-4,-2,0,0,0)

fit_tam <- function(d) {
    persons <- sort(unique(d$Person))
    i <- match(d$Person,persons)
    j <- match(keys(d),keys(cells))
    stopifnot(length(persons)==120L,nrow(d)==480L,!anyDuplicated(data.frame(i,j)))
    y <- matrix(NA_integer_,length(persons),24,dimnames=list(persons,keys(cells)))
    y[cbind(i,j)] <- d$Score-1L
    fit <- TAM::tam.mml(resp=y,A=A,B=B,beta.fixed=matrix(c(1,1,0),1),variance.fixed=matrix(c(1,1,1),1),
        verbose=FALSE,control=list(nodes=nodes,conv=1e-10,convD=1e-10,convM=1e-10,maxiter=2000))
    x <- fit$xsi$xsi
    counts <- matrix(0,length(persons),120)
    counts[cbind(i,j+24*(d$Score-1L))] <- 1
    nll <- function(u) {
        p <- TAM:::tam_calc_prob(1:24,A,matrix(0,24,5),B,u,matrix(nodes),161L,5L)$rprobs
        -sum(log(exp(counts%*%matrix(log(p),120,161))%*%weights))
    }
    stopifnot(fit$iter<2000L,abs(nll(x)-fit$deviance/2)<1e-7)
    h <- optimHess(x,nll,control=list(ndeps=rep(2e-4,12)))
    v <- chol2inv(chol(h))
    data.frame(route="TAM joint",metric="severity logit",estimate=sum(joint_weight*x),
        se=sqrt(drop(crossprod(joint_weight,v%*%joint_weight))),status="verified")
}

fit_mfrmr <- function(d) {
    setup <- list(data=d,person="Person",facets=c("Item","Rater"),score="Score",model="RSM",method="MML",
        rating_min=1,rating_max=5,keep_original=TRUE,quad_points=61L,reltol=1e-12)
    joint <- suppressWarnings(do.call(mfrmr::fit_mfrm,modifyList(setup,list(facets=c("Item","Rater","School"),
        noncenter_facet="School",positive_facets="School",facet_interactions="Rater:School"))))
    term <- names(joint$config$interaction_specs)
    effects <- joint$interactions$effects
    stopifnot(all(effects$FacetA=="Rater"),all(effects$FacetB=="School"))
    label <- with(effects,paste0("Rater=",FacetA_Level," | School=",FacetB_Level))
    weight <- c("Rater=R3 | School=B"=-1,"Rater=R1 | School=B"=1,
                "Rater=R3 | School=A"=1,"Rater=R1 | School=A"=-1)
    w <- ifelse(label %in% names(weight),weight[label],0)
    L <- matrix(w,1,dimnames=list("DFF severity",label))
    ci <- suppressWarnings(mfrmr::mfrm_facet_intervals(joint,interaction=term,contrasts=L))
    cell <- ci$table
    stopifnot(abs(cell$Estimate-sum(effects$Estimate*w))<1e-10)
    checks <- ci$checks
    failed <- if (is.null(checks)) "" else paste(checks$Check[!checks$Passed],collapse=" | ")
    joint_row <- data.frame(route="mfrmr joint",metric="severity logit",estimate=cell$Estimate,
                            se=cell$SE,status=cell$Status,detail=failed)

    pooled <- suppressWarnings(do.call(mfrmr::fit_mfrm,c(setup,list(min_common_anchors=3L))))
    diagnostics <- suppressWarnings(mfrmr::diagnose_mfrm(pooled))
    other <- lapply(c("residual","refit"),function(mode) {
        dff <- suppressWarnings(mfrmr::analyze_dff(pooled,diagnostics,facet="Rater",group="School",data=d,method=mode))
        t <- dff$dif_table[match(c("R3","R1"),dff$dif_table$Level),]
        stopifnot(all(t$Group1=="A"),all(t$Group2=="B"),!any(t$SupportsFormalInference))
        # Residuals measure higher scores; refits measure higher severity.
        value <- diff(rev(t$Contrast))*if (mode=="refit") -1 else 1
        data.frame(route=paste("mfrmr",mode),metric=if (mode=="refit") "severity logit" else "score residual A-B",
                   estimate=value,se=NA_real_,status=paste(unique(t$ScaleLinkStatus),collapse=" | "),
                   detail=paste(unique(t$SEBasis),collapse=" | "))
    })
    do.call(rbind,c(list(joint_row),other))
}

parts <- split(data,list(data$replicate,data$design,data$effect),drop=TRUE)
out <- lapply(parts,function(d) {
    tam <- fit_tam(d)
    tam$detail <- "Full observed information from the TAM marginal likelihood"
    table <- rbind(tam,fit_mfrmr(d))
    cbind(replicate=d$replicate[1],design=d$design[1],effect=d$effect[1],table)
})
write.csv(do.call(rbind,out),args[2],row.names=FALSE)
