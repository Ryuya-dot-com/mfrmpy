# Independent fixed-SD marginal fit and covariance for the same-mode contrast.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, packageVersion("TAM")=="4.3.25")
ratings <- read.csv(args[1])
nodes <- seq(-8,8,length.out=161)
weights <- dnorm(nodes)/sum(dnorm(nodes))

fit <- function(d, model) {
    rater <- if(model=="common_mode") "RaterYear" else "RaterYearMode"
    target <- if(model=="common_mode") c("A-R3","B-R3") else c("A-R3-video","B-R3-video")
    if(!all(target %in% d[[rater]]))
        return(data.frame(model, estimate=NA_real_, se=NA_real_))
    endpoint <- tapply(d$Score,d[[rater]],function(y) all(y==1) || all(y==5))
    if(any(endpoint))
        return(data.frame(model, estimate=NA_real_, se=NA_real_))
    fields <- c("WorkYear",rater,if(model=="common_mode") "Mode")
    cells <- unique(d[fields])
    key <- function(dat) do.call(paste,dat[fields])
    levels <- sort(unique(cells[[rater]]))
    location <- cbind(ifelse(cells$WorkYear=="A",1,-1),
                      -diag(length(levels))[match(cells[[rater]],levels),])
    if(model=="common_mode") location <- cbind(location,ifelse(cells$Mode=="live",-1,1))
    design <- array(0,c(nrow(cells),5,ncol(location)+3))
    for(k in 1:4) {
        design[,k+1,seq_len(ncol(location))] <- k*location
        design[,k+1,ncol(location)+1:3] <- rep(-colSums(rbind(diag(3),-1)[seq_len(k),,drop=FALSE]),each=nrow(cells))
    }
    slope <- array(rep(0:4,each=nrow(cells)),c(nrow(cells),5,1))
    person <- match(d$Recording,sort(unique(d$Recording)))
    item <- match(key(d),key(cells))
    y <- matrix(NA_integer_,max(person),nrow(cells))
    stopifnot(nrow(d)==264L,!anyDuplicated(data.frame(person,item)))
    y[cbind(person,item)] <- d$Score-1L
    res <- TAM::tam.mml(y,A=design,B=slope,beta.fixed=matrix(c(1,1,0),1),variance.fixed=matrix(c(1,1,1),1),
                       verbose=FALSE,control=list(nodes=nodes,conv=1e-10,convD=1e-10,convM=1e-10,maxiter=4000))
    n <- nrow(cells)
    counts <- matrix(0,nrow(y),n*5)
    counts[cbind(person,item+n*(d$Score-1L))] <- 1
    nll <- function(u) {
        p <- TAM:::tam_calc_prob(seq_len(n),design,matrix(0,n,5),slope,u,matrix(nodes),length(nodes),5L)$rprobs
        -sum(log(exp(counts%*%matrix(log(p),n*5,length(nodes)))%*%weights))
    }
    point <- res$xsi$xsi
    stopifnot(res$iter<4000L,abs(nll(point)-res$deviance/2)<1e-7)
    cov <- chol2inv(chol(optimHess(point,nll,control=list(ndeps=rep(2e-4,length(point))))))
    contrast <- numeric(length(point))
    contrast[1L+match(target,levels)] <- c(-1,1)
    data.frame(model,estimate=sum(contrast*point),se=sqrt(drop(contrast%*%cov%*%contrast)))
}

sets <- split(ratings,list(ratings$replicate,ratings$case,ratings$allocation),drop=TRUE)
out <- lapply(sets,function(d) {
    stats <- rbind(fit(d,"common_mode"),fit(d,"separate_cells"))
    cbind(d[rep(1L,2L),c("replicate","case","allocation")],stats)
})
write.csv(do.call(rbind,out),args[2],row.names=FALSE)
