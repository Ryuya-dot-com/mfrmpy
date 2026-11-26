args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1L], .libPaths()))
library(mfrmr)
stopifnot(as.character(packageVersion("mfrmr")) == "0.3.2")
results <- lapply(c("one", "two"), function(name) {
  data <- read.csv(file.path(args[2L], paste0("gd_interval_", name, ".csv")))
  facets <- intersect(c("Rater", "Task"), names(data))
  g <- mfrm_g_study(data, person = "Person", facets = facets, score = "Score", method = "urgenova")
  list(name = name, components = g$variance_components,
       covariance = unclass(vcov(g)), covariance_names = rownames(vcov(g)),
       intervals = unclass(confint(g, component = "coefficients")),
       interval_names = c("G", "Phi"))
})
jsonlite::write_json(results, file.path(args[2L], "gd_interval_mfrmr.json"), digits = NA, auto_unbox = TRUE)
