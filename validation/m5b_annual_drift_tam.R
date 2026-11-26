# Independent TAM check of the free and old-anchor annual contrasts.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, packageVersion("TAM")=="4.3.25")
ratings <- read.csv(args[1])
nodes <- seq(-8,8,length.out=161)
weights <- dnorm(nodes)/sum(dnorm(nodes))
identity <- diag(10)
anchors <- cbind(identity[,1],rowSums(identity[,c(2,3,5,6)]),identity[,c(4,7:10)],.6*(identity[,3]+identity[,6]))
drift <- identity[5:7,]-identity[2:4,]
target <- rbind(R3=drift[3,],panel_mean=colMeans(drift),R3_relative=drift[3,]-colMeans(drift[1:2,]))
labels <- c(paste0("A-R",1:3),paste0("B-R",1:3))
step_basis <- rbind(diag(3),-1)

fit <- function(d, anchor) {
    cells <- unique(d[c("WorkYear","RaterYear")])
    key <- function(dat) with(dat,paste(WorkYear,RaterYear))
    loc <- cbind(ifelse(cells$WorkYear=="A",1,-1),-diag(6)[match(cells$RaterYear,labels),])
    n <- nrow(cells)
    A_raw <- array(0,c(n,5,10))
    for (k in 1:4) {
        A_raw[,k+1,1:7] <- k*loc
        A_raw[,k+1,8:10] <- -rep(colSums(step_basis[seq_len(k),,drop=FALSE]),each=n)
    }
    projection <- if (anchor) anchors else identity
    dimension <- ncol(projection)
    free <- seq_len(dimension-as.integer(anchor))
    A <- array(matrix(A_raw,n*5,10)%*%projection,c(n,5,dimension),
               dimnames=list(key(cells),as.character(0:4),paste0("p",seq_len(dimension))))
    B <- array(rep(0:4,each=n),c(n,5,1))
    ids <- sort(unique(d$Recording))
    person <- match(d$Recording,ids)
    item <- match(key(d),key(cells))
    y <- matrix(NA_integer_,length(ids),n,dimnames=list(ids,key(cells)))
    stopifnot(nrow(d)==240L,!anyDuplicated(data.frame(person,item)))
    y[cbind(person,item)] <- d$Score-1L
    fitted <- TAM::tam.mml(resp=y,A=A,B=B,xsi.fixed=if(anchor) matrix(c(dimension,1),1) else NULL,
        variance.fixed=matrix(c(1,1,1),1),beta.fixed=matrix(c(1,1,0),1),verbose=FALSE,
        control=list(nodes=nodes,conv=1e-10,convD=1e-10,convM=1e-10,maxiter=3000))
    x <- fitted$xsi$xsi
    counts <- matrix(0,length(ids),n*5)
    counts[cbind(person,item+n*(d$Score-1L))] <- 1
    nll <- function(u) {
        point <- x
        point[free] <- u
        p <- TAM:::tam_calc_prob(seq_len(n),A,matrix(0,n,5),B,point,matrix(nodes),161L,5L)$rprobs
        -sum(log(exp(counts%*%matrix(log(p),n*5,161))%*%weights))
    }
    stopifnot(fitted$iter<3000L,abs(nll(x[free])-fitted$deviance/2)<1e-7,!anchor||x[dimension]==1)
    h <- optimHess(x[free],nll,control=list(ndeps=rep(2e-4,length(free))))
    v <- chol2inv(chol(h))
    L <- target%*%projection
    cov <- L[,free]%*%v%*%t(L[,free])
    data.frame(model=if(anchor) "old_anchors" else "recordings",target=rownames(target),
               estimate=drop(L%*%x),se=sqrt(diag(cov)))
}

parts <- split(ratings,list(ratings$replicate,ratings$design,ratings$effect),drop=TRUE)
out <- lapply(parts,function(d) {
    cbind(replicate=d$replicate[1],design=d$design[1],effect=d$effect[1],
          rbind(fit(d,FALSE),fit(d,TRUE)))
})
write.csv(do.call(rbind,out),args[2],row.names=FALSE)
