# Check shared and separate step parameters through independent TAM arrays.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% 2:3, packageVersion("TAM")=="4.3.25")
ratings <- read.csv(args[1])
labels <- c(paste0("A-R",1:3),paste0("B-R",1:3))
models <- c("RSM","PCM_shared_except_B_R3","PCM_RaterYear")
if (length(args)==3L) {
    selected <- strsplit(args[3],",",fixed=TRUE)[[1]]
    stopifnot(all(selected %in% models),!anyDuplicated(selected))
    models <- selected
}
nodes <- seq(-8,8,length.out=161)
weights <- dnorm(nodes)/sum(dnorm(nodes))
step_basis <- rbind(diag(3),-1)
step_width <- rbind(width3=c(0,-1,1),width4=c(-1,-1,-2))
key <- function(d) with(d,paste(WorkYear,RaterYear))

fit <- function(d, model) {
    group <- switch(model,RSM=rep("all",nrow(d)),PCM_shared_except_B_R3=d$StepProfile,PCM_RaterYear=d$RaterYear)
    owners <- sort(unique(group))
    if (any(table(group,factor(d$Score,levels=1:5))==0L))
        return(data.frame(target="severity",estimate=NA_real_,se=NA_real_,reason="unobserved score in a step set"))
    cells <- unique(d[c("WorkYear","RaterYear")])
    owner <- match(group[match(key(cells),key(d))],owners)
    n <- nrow(cells)
    dimension <- 7L+3L*length(owners)
    locations <- cbind(ifelse(cells$WorkYear=="A",1,-1),-diag(6)[match(cells$RaterYear,labels),])
    A <- array(0,c(n,5,dimension),dimnames=list(key(cells),as.character(0:4),paste0("p",seq_len(dimension))))
    B <- array(rep(0:4,each=n),c(n,5,1))
    for (k in 1:4) {
        A[,k+1,1:7] <- k*locations
        step <- -colSums(step_basis[seq_len(k),,drop=FALSE])
        for (g in seq_along(owners)) {
            active <- owner==g
            A[active,k+1,7L+3L*(g-1L)+1:3] <- rep(step,each=sum(active))
        }
    }
    person <- match(d$Recording,sort(unique(d$Recording)))
    item <- match(key(d),key(cells))
    y <- matrix(NA_integer_,max(person),n,dimnames=list(NULL,key(cells)))
    stopifnot(nrow(d)>0L,!anyDuplicated(data.frame(person,item)))
    y[cbind(person,item)] <- d$Score-1L
    out <- TAM::tam.mml(y,A=A,B=B,beta.fixed=matrix(c(1,1,0),1),variance.fixed=matrix(c(1,1,1),1),
        verbose=FALSE,control=list(nodes=nodes,conv=1e-10,convD=1e-10,convM=1e-10,maxiter=3000))
    x <- out$xsi$xsi
    counts <- matrix(0,nrow(y),n*5)
    counts[cbind(person,item+n*(d$Score-1L))] <- 1
    nll <- function(u) {
        p <- TAM:::tam_calc_prob(seq_len(n),A,matrix(0,n,5),B,u,matrix(nodes),161L,5L)$rprobs
        -sum(log(exp(counts%*%matrix(log(p),n*5,161))%*%weights))
    }
    stopifnot(out$iter<3000L,abs(nll(x)-out$deviance/2)<1e-7)
    covariance <- chol2inv(chol(optimHess(x,nll,control=list(ndeps=rep(2e-4,dimension)))))
    L <- matrix(0,if (model=="RSM") 1 else 3,dimension)
    L[1,c(4,7)] <- c(-1,1)
    if (nrow(L)==3L) {
        owners_ab <- if (model=="PCM_RaterYear") c("A-R3","B-R3") else c("common","separate")
        L[2:3,7L+3L*(match(owners_ab[1],owners)-1L)+1:3] <- -step_width
        L[2:3,7L+3L*(match(owners_ab[2],owners)-1L)+1:3] <- step_width
    }
    data.frame(target=c("severity","width3","width4")[seq_len(nrow(L))],
               estimate=drop(L%*%x),se=sqrt(diag(L%*%covariance%*%t(L))),reason="")
}

sets <- split(ratings,list(ratings$replicate,ratings$condition),drop=TRUE)
fits <- lapply(sets,function(d) do.call(rbind,lapply(models,function(m)
    cbind(replicate=d$replicate[1],condition=d$condition[1],model=m,fit(d,m)))))
write.csv(do.call(rbind,fits),args[2],row.names=FALSE)
