# Independent calibration and receiving-sample intervals with pinned mfrmr 0.3.0.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
reps <- seq.int(as.integer(args[3]), as.integer(args[4]))
designs <- c("balanced6", "balanced24")
items <- paste0("I", 1:4)
raters <- paste0("R", 1:3)
anchor_levels <- raters[1:2]
contrast <- matrix(c(-1, 0, 1), 1, dimnames = list("R3_minus_R1", raters))

generate_source <- function(rep) {
    set.seed(2026112100L + rep, kind = "L'Ecuyer-CMRG", normal.kind = "Inversion", sample.kind = "Rejection")
    people <- sprintf("S%03d", 1:60)
    data <- expand.grid(Person = people, Item = items, Rater = raters, stringsAsFactors = FALSE)
    eta <- rnorm(60)[match(data$Person, people)] - c(-.75, -.25, .25, .75)[match(data$Item, items)] -
        c(-.6, 0, .6)[match(data$Rater, raters)]
    w <- exp(sweep(outer(eta, 0:3), 2, cumsum(c(0, -.9, 0, .9)), "-"))
    cutoffs <- t(apply(w / rowSums(w), 1, cumsum))[, 1:3]
    data$Score <- rowSums(runif(nrow(data)) > cutoffs)
    data
}

if (identical(args[1], "--calibration-only")) {
    stopifnot(!file.exists(args[2]))
    source <- do.call(rbind, lapply(reps, function(rep) transform(generate_source(rep), replicate = rep)))
    write.csv(source, args[2], row.names = FALSE)
    quit(save = "no")
}

stopifnot(packageVersion("mfrmr") == "0.3.0")
ratings <- read.csv(args[1])
ratings <- ratings[ratings$design %in% designs & ratings$replicate %in% reps, ]
targets <- split(ratings, list(ratings$replicate, ratings$design), drop = TRUE)

estimate <- function(data, anchors = NULL) {
    mfrmr::fit_mfrm(data, "Person", c("Item", "Rater"), "Score", model = "RSM", method = "MML",
        rating_min = 0, rating_max = 3, anchor_policy = "silent",
        spec = if (is.null(anchors)) list() else list(anchors = anchors, noncenter_facet = "Rater",
            population_formula = ~1, person_data = unique(data["Person"]), population_sd = 1),
        control = list(mml_engine = "direct", mml_integration = "fixed", quad_points = 31L, reltol = 1e-12))
}

effects <- function(fit, facet, levels) {
    tab <- subset(fit$facets$others, Facet == facet)
    tab$Estimate[match(levels, tab$Level)]
}

source_fit <- function(data) {
    fit <- estimate(data)
    ci <- mfrmr::mfrm_facet_intervals(fit, "Rater")
    tab <- ci$table[match(anchor_levels, ci$table$Target), ]
    covariance <- ci$covariance[anchor_levels, anchor_levels, drop = FALSE]
    if (any(!is.finite(covariance))) stop(paste(tab$Status, tab$InferenceReview, collapse = " | "))
    list(anchors = data.frame(Facet = "Rater", Level = anchor_levels, Anchor = tab$Estimate),
         covariance = covariance)
}

receiving_fit <- function(data, source) {
    values <- source$anchors$Anchor
    v <- source$covariance
    out <- list(a1 = values[1], a2 = values[2], v11 = v[1, 1], v12 = v[1, 2], v22 = v[2, 2])
    fit <- tryCatch(estimate(data, source$anchors), error = identity)
    if (inherits(fit, "error")) return(c(out, list(reason = conditionMessage(fit))))
    r <- effects(fit, "Rater", raters)
    stopifnot(all(abs(r[1:2] - values) < 1e-12), nrow(fit$steps) == 3L)
    out <- c(out, as.list(setNames(effects(fit, "Item", items)[1:3], items[1:3])),
             list(step1 = fit$steps$Estimate[1], step2 = fit$steps$Estimate[2], r3 = r[3],
                  mean = unname(fit$population$coefficients[1]), estimate = r[3] - r[1]))
    interval <- function(covariance) mfrmr::mfrm_facet_intervals(fit, "Rater", contrasts = contrast,
        anchor_covariance = covariance, independent_anchors = TRUE)
    ci <- tryCatch(interval(v), error = identity)
    if (inherits(ci, "error")) return(c(out, list(reason = conditionMessage(ci))))
    tab <- ci$table
    if (!is.finite(tab$SE)) return(c(out, list(reason = paste(tab$Status, tab$InferenceReview,
        paste(ci$anchor_uncertainty$checks$Check[!ci$anchor_uncertainty$checks$Passed], collapse = " | ")))))
    zero <- interval(0 * v)$table
    stopifnot(nrow(tab) == 1L, abs(tab$Estimate - out$estimate) < 1e-12, tab$SE >= tab$ConditionalSE,
        max(abs(c(zero$SE, zero$Lower, zero$Upper) - c(tab$ConditionalSE, tab$ConditionalLower,
            tab$ConditionalUpper))) < 1e-12)
    jac <- ci$anchor_uncertainty$target_sensitivity[1, anchor_levels]
    stopifnot(abs(sum(jac)) < 1e-5)
    c(out, list(conditional_se = tab$ConditionalSE, conditional_lower = tab$ConditionalLower,
        conditional_upper = tab$ConditionalUpper, propagated_se = tab$SE, propagated_lower = tab$Lower,
        propagated_upper = tab$Upper, anchor_se = tab$AnchorSourceSE, j1 = unname(jac[1]), j2 = unname(jac[2]),
        zero_covariance_checked = TRUE, reason = ""))
}

fields <- c("a1", "a2", "v11", "v12", "v22", items[1:3], "step1", "step2", "r3", "mean", "estimate",
    "conditional_se", "conditional_lower", "conditional_upper", "propagated_se", "propagated_lower",
    "propagated_upper", "anchor_se", "j1", "j2", "zero_covariance_checked", "reason")
stopifnot(!file.exists(args[2]))
for (rep in reps) {
    source_data <- generate_source(rep)
    source <- tryCatch(suppressWarnings(source_fit(source_data)), error = identity)
    rows <- lapply(designs, function(design) {
        target <- targets[[paste(rep, design, sep = ".")]]
        stopifnot(nrow(target) == 240L, length(unique(target$Person)) == 60L,
                  !any(target$Person %in% source_data$Person))
        out <- if (inherits(source, "error")) list(reason = paste("Calibration:", conditionMessage(source))) else
            suppressWarnings(receiving_fit(target, source))
        for (name in setdiff(fields, names(out))) out[[name]] <- NA
        data.frame(replicate = rep, design = design, out[fields], check.names = FALSE)
    })
    write.table(do.call(rbind, rows), args[2], sep = ",", row.names = FALSE, na = "",
                append = rep != reps[1], col.names = rep == reps[1])
}
