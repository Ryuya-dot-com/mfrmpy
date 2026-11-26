# Run with Rscript --vanilla validation/tam_category_design.R (TAM 4.3.25).
# Free-coordinate counterpart to the expanded terms in test_category_design_probabilities_and_severity_contrasts.
cells <- expand.grid(rater = 0:1, task = 0:1)
k <- 0:2
ability <- log(2)
for (model in c("RSM", "PCM")) {
    widths <- if (model == "RSM") log(2) else log(c(2, 3))
    for (joint in c(FALSE, TRUE)) {
        # R0 is free, R1 is derived. Both task difficulties are free.
        # Each step pair is (-width, +width). A joint model has one free cell.
        xi <- c(ability, ability, -ability, widths, if (joint) ability)
        A <- array(0, c(4, 3, length(xi)))
        B <- array(rep(k, each = 4), c(4, 3, 1))
        expected <- matrix(0, 4, 3)
        for (i in seq_len(nrow(cells))) {
            r <- cells$rater[i]
            t <- cells$task[i]
            width_id <- if (model == "RSM") 1L else t + 1L
            sign <- (1 - 2*r) * (1 - 2*t)
            A[i, , 1] <- k * (1 - 2*r)
            A[i, , 2 + t] <- -k
            A[i, , 3 + width_id] <- c(0, 1, 0)
            if (joint) A[i, , length(xi)] <- k * sign
            eta <- ability + (1 - 2*r)*ability - (1 - 2*t)*ability + as.integer(joint)*sign*ability
            weights <- c(1, exp(eta + widths[width_id]), exp(2*eta))
            expected[i, ] <- weights / sum(weights)
        }
        # Evaluate TAM's category kernel at fixed coefficients without fitting a population.
        probs <- TAM:::tam_calc_prob(1:4, A, matrix(0, 4, 3), B, xi, matrix(ability),
                                     nnodes = 1L, maxK = 3L)$rprobs[, , 1]
        error <- max(abs(probs - expected))
        severity_difference <- -sum(c(1, -1, -1, 1) * log(probs[, 2] / probs[, 1]))
        stopifnot(error < 1e-13, abs(severity_difference + as.integer(joint)*4*ability) < 1e-13)
        cat(sprintf("%s joint=%s   max probability error %.3g   severity contrast %.12f\n",
                    model, joint, error, severity_difference))
    }
}
