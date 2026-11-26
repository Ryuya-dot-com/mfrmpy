# Compare TAM's printed SE and full marginal information at separate TAM fits.
suppressPackageStartupMessages(library(TAM))

cells <- expand.grid(Item = paste0("I", 1:4), Rater = paste0("R", 1:3))
ids <- paste(cells$Item, cells$Rater)
items <- rbind(diag(3), -1)[match(cells$Item, paste0("I", 1:4)), ]
raters <- match(cells$Rater, paste0("R", 1:3))
steps <- rbind(diag(2), -1)
source_design <- cbind(items, rbind(diag(2), -1)[raters, ])
target_design <- cbind(items, 1, raters == 2, raters == 3)
B <- array(rep(0:3, each = 12), c(12, 4, 1), dimnames = list(ids, as.character(0:3), "ability"))

estimate <- function(data, locations, fixed = NULL, sd = 1) {
    nodes <- seq(-8, 8, length.out = 161)*sd
    weights <- dnorm(nodes, sd = sd) / sum(dnorm(nodes, sd = sd))
    ncoef <- ncol(locations) + 2L
    A <- array(0, c(12, 4, ncoef))
    for (k in 1:3) {
        A[, k+1, seq_len(ncoef-2)] <- -k*locations
        A[, k+1, (ncoef-1):ncoef] <- -rep(colSums(steps[seq_len(k), , drop = FALSE]), each = 12)
    }
    person <- sort(unique(data$Person))
    p <- match(data$Person, person)
    j <- match(paste(data$Item, data$Rater), ids)
    stopifnot(length(person) == 60L, !anyDuplicated(data.frame(p, j)))
    y <- matrix(NA_real_, length(person), length(ids), dimnames = list(person, ids))
    y[cbind(p, j)] <- data$Score
    fixed_idx <- if (is.null(fixed)) integer() else fixed[1, 1]
    free <- setdiff(seq_len(ncoef), fixed_idx)
    fit <- tam.mml(resp = y, A = A, B = B, xsi.fixed = fixed, variance.fixed = matrix(c(1, 1, sd^2), 1),
        beta.fixed = matrix(c(1, 1, 0), 1), verbose = FALSE,
        control = list(nodes = nodes, conv = 1e-10, convD = 1e-10, convM = 1e-10, maxiter = 2000))
    point <- fit$xsi$xsi
    stopifnot(fit$iter < 2000L, length(point) == ncoef, all(is.finite(point)),
        !length(fixed_idx) || abs(point[fixed_idx] - fixed[1, 2]) < 1e-12)
    counts <- matrix(0, length(person), 48)
    counts[cbind(p, j + 12*data$Score)] <- 1
    nll <- function(x) {
        probs <- TAM:::tam_calc_prob(1:12, A, matrix(0, 12, 4), B, x, matrix(nodes), 161L, 4L)$rprobs
        -sum(log(exp(counts %*% matrix(log(probs), 48, 161)) %*% weights))
    }
    stopifnot(abs(nll(point) - fit$deviance/2) < 1e-7)
    h <- optimHess(point, nll, control = list(ndeps = rep(2e-4, ncoef)))
    cov <- chol2inv(chol(h[free, free]))
    delta <- diag(2e-4, ncoef)[free, , drop = FALSE]
    gradient <- apply(delta, 1, function(d) (nll(point+d) - nll(point-d))/4e-4)
    stopifnot(sqrt(drop(crossprod(gradient, cov %*% gradient))) <= .001)
    list(point = point, cov = cov, free = free, h = h, printed = fit$xsi$se.xsi, fit = fit)
}

if (sys.nframe() == 0L) {
    args <- commandArgs(trailingOnly = TRUE)
    stopifnot(length(args) == 4L, packageVersion("TAM") == "4.3.25", !file.exists(args[4]))
    reference <- read.csv(gzfile(args[1]))
    calibrations <- read.csv(args[2])
    calibrations <- split(calibrations, calibrations$replicate)
    targets <- read.csv(args[3])
    targets <- targets[targets$replicate %in% names(calibrations) & targets$design %in% unique(reference$design), ]
    targets <- split(targets, list(targets$replicate, targets$design), drop = TRUE)

    rows <- list()
    for (rep in as.integer(names(calibrations))) {
        src <- estimate(calibrations[[as.character(rep)]], source_design)
        for (design in unique(reference$design)) {
            ref <- reference[reference$replicate == rep & reference$design == design, ]
            data <- targets[[paste(rep, design, sep = ".")]]
            stopifnot(nrow(ref) == 1L, nrow(data) == 240L, nrow(calibrations[[as.character(rep)]]) == 720L)
            fit <- estimate(data, target_design, fixed = matrix(c(5, ref$a2 - ref$a1), 1))
            at <- match(6, fit$free)
            d <- drop(-fit$cov[at, ] %*% fit$h[fit$free, 5])
            cond_se <- sqrt(fit$cov[at, at])
            anchor_se <- abs(d)*sqrt(sum(src$cov[4:5, 4:5]*matrix(c(1,-1,-1,1), 2)))
            invisible(capture.output(quick <- TAM::tam.se(fit$fit)))
            row <- data.frame(engine = "TAM", replicate = rep, design = design, a1 = src$point[4],
                a2 = src$point[5], v11 = src$cov[4,4], v12 = src$cov[4,5], v22 = src$cov[5,5],
                estimate = fit$point[6], conditional_se = cond_se, j2 = d, anchor_se = anchor_se,
                propagated_se = sqrt(cond_se^2 + anchor_se^2), printed_se = fit$printed[6],
                quick_se = quick$xsi$se[6], reason = "")
            rows[[length(rows)+1L]] <- row
        }
    }
    write.csv(do.call(rbind, rows), args[4], row.names = FALSE)
}
