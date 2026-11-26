args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1L], .libPaths()))
library(mfrmr)
stopifnot(as.character(packageVersion("mfrmr")) == "0.3.2")
scores <- c("Content", "Language", "Style")
weights <- c(Content = .2, Language = .5, Style = .3)
results <- lapply(c("one", "two"), function(name) {
  data <- read.csv(file.path(args[2L], paste0("multivariate_gd_", name, ".csv")))
  facets <- intersect(c("Rater", "Task"), names(data))
  grid <- as.data.frame(setNames(lapply(facets, function(f) c(length(unique(data[[f]])), 13)), facets))
  anova <- mfrm_g_study(data, scores = scores, person = "Person", facets = facets, method = "anova")
  h <- mfrm_g_study(data, scores = scores, person = "Person", facets = facets,
                    weights = weights, d_study_grid = grid, method = "urgenova")
  list(name = name, scores = scores, weights = weights, components = anova$components,
       component_diagnostics = anova$component_diagnostics,
       composite_components = h$variance_components, univariate = h$univariate,
       observed = h$g_study_summary, plans = h$d_study)
})
jsonlite::write_json(results, file.path(args[2L], "multivariate_gd_mfrmr.json"), digits = NA, auto_unbox = TRUE)
