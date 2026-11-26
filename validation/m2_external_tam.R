# TAM 4.3.25. Run with the same ratings.csv used by m2_external_conquest.py.
suppressPackageStartupMessages(library(TAM))
args <- commandArgs(trailingOnly = TRUE)
ratings <- read.csv(args[1])
destination <- args[2]
workers <- as.integer(args[3])
stopifnot(as.character(packageVersion("TAM")) == "4.3.25")

cells <- expand.grid(Item = paste0("I", 1:4), Rater = paste0("R", 1:3))
ids <- paste(cells$Item, cells$Rater)
pars <- c("item I1", "item I2", "item I3", "rater R1", "rater R2", "category 1", "category 2")
C_item <- rbind(diag(3), c(-1, -1, -1))
C_rater <- rbind(diag(2), c(-1, -1))
A <- array(0, c(12, 4, 7), dimnames = list(ids, as.character(0:3), pars))
B <- array(rep(0:3, each = 12), c(12, 4, 1))
for (j in 1:12) for (k in 1:3) {
    A[j, k+1, 1:3] <- -k*C_item[match(cells$Item[j], paste0("I", 1:4)), ]
    A[j, k+1, 4:5] <- -k*C_rater[match(cells$Rater[j], paste0("R", 1:3)), ]
    A[j, k+1, 6:7] <- -colSums(C_rater[seq_len(k), , drop = FALSE])
}
nodes <- seq(-8, 8, length.out = 161)
weights <- dnorm(nodes)/sum(dnorm(nodes))
contrast <- c(0, 0, 0, -2, -1, 0, 0)

estimate <- function(d) {
    out <- data.frame(replicate=d$replicate[1], design=d$design[1], estimate=NA_real_, se=NA_real_,
                      lower=NA_real_, upper=NA_real_, deviance=NA_real_, reason="", scaled_gradient=NA_real_)
    for (name in pars) out[[name]] <- NA_real_
    tryCatch({
        person <- sort(unique(d$Person))
        p <- match(d$Person, person)
        j <- match(paste(d$Item, d$Rater), ids)
        y <- matrix(NA_real_, length(person), length(ids), dimnames=list(person, ids))
        y[cbind(p, j)] <- d$Score
        stopifnot(nrow(y) == 60, sum(!is.na(y)) == 240)
        fit <- tam.mml(resp=y, A=A, B=B, variance.fixed=matrix(c(1,1,1), 1, 3),
                       beta.fixed=matrix(c(1,1,0), 1, 3), verbose=FALSE,
                       control=list(nodes=nodes, conv=1e-10, convD=1e-10, convM=1e-10, maxiter=2000))
        xi <- fit$xsi$xsi
        stopifnot(length(xi) == 7, all(is.finite(xi)), fit$iter<2000)

        # Differentiate the integrated response likelihood, retaining the full covariance.
        counts <- matrix(0, nrow(y), 48)
        counts[cbind(p, j + 12*d$Score)] <- 1
        nll <- function(x) {
            rprobs <- TAM:::tam_calc_prob(1:12, A, matrix(0,12,4), B, x, matrix(nodes), 161L, 4L)$rprobs
            ll <- counts %*% matrix(log(rprobs), 48, 161)
            -sum(log(exp(ll) %*% weights))
        }
        stopifnot(abs(nll(xi) - fit$deviance/2) < 1e-7)
        h <- optimHess(xi, nll, control=list(ndeps=rep(0.0002, 7)))
        stopifnot(min(eigen(h, symmetric=TRUE, only.values=TRUE)$values) > 0)
        cov <- chol2inv(chol(h))
        directions <- diag(0.0002, 7)
        grad <- apply(directions, 1, function(delta) (nll(xi+delta)-nll(xi-delta)) / 0.0004)
        out$scaled_gradient <- sqrt(drop(crossprod(grad, cov %*% grad)))
        stopifnot(out$scaled_gradient <= .01)
        out$deviance <- fit$deviance
        out$estimate <- sum(contrast*xi)
        out$se <- sqrt(drop(crossprod(contrast, cov %*% contrast)))
        out$lower <- out$estimate - qnorm(.975)*out$se
        out$upper <- out$estimate + qnorm(.975)*out$se
        for (i in seq_along(pars)) out[[pars[i]]] <- xi[i]
        out
    }, error=function(exc) {
        out$reason <- conditionMessage(exc)
        out
    })
}

key <- function(d) paste(d$replicate, d$design)
if (file.exists(destination)) {
    finished <- read.csv(destination)
    ratings <- ratings[!key(ratings) %in% key(finished), ]
}
groups <- split(ratings, key(ratings))
for (batch in split(seq_along(groups), ceiling(seq_along(groups)/100))) {
    rows <- parallel::mclapply(groups[batch], estimate, mc.cores=workers)
    out <- do.call(rbind, rows)
    append <- file.exists(destination)
    write.table(out, destination, sep=",", col.names=!append, row.names=FALSE, append=append, na="")
    cat(max(batch), "/", length(groups), " TAM calibrations\n", sep="")
}
