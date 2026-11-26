args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[1L], .libPaths()))
library(mfrmr)
stopifnot(as.character(packageVersion('mfrmr')) == '0.3.2')
results <- lapply(c('one', 'crossed', 'nested_balanced', 'nested_unequal', 'nested_missing'), function(name) {
  d <- read.csv(file.path(args[2L], paste0('unbalanced_gd_', name, '.csv')))
  facets <- intersect(c('Rater', 'Task'), names(d))
  nested <- startsWith(name, 'nested')
  fit <- mfrm_g_study(d, scores=c('A', 'B'), person='Person', facets=facets,
                      method='minque0', missing='omit', nesting=if (nested) c(Rater='Task') else NULL)
  grid <- as.data.frame(setNames(lapply(facets, function(f) if (f=='Rater') 5 else 4), facets))
  weights <- c(A=.3, B=.7)
  raw <- mfrm_d_study(fit, design_grid=grid, weights=weights, negative_components='raw')
  vectors <- list(A=c(1,0), B=c(0,1), Composite=weights)
  clipped <- lapply(names(vectors), function(target) {
    w <- vectors[[target]]
    v <- vapply(fit$components, function(s) max(0, drop(t(w) %*% s %*% w)), numeric(1))
    if (nested) {
      rel <- v['Person:Task']/4 + v['Residual']/20
      abs <- rel + v['Task']/4 + v['Rater(Task)']/20
    } else if (length(facets)==1L) {
      rel <- v['Residual']/5
      abs <- rel + v['Rater']/5
    } else {
      rel <- v['Person:Rater']/5 + v['Person:Task']/4 + v['Residual']/20
      abs <- rel + v['Rater']/5 + v['Task']/4 + v['Rater:Task']/20
    }
    list(Target=target, G=unname(v['Person']/(v['Person']+rel)), Phi=unname(v['Person']/(v['Person']+abs)),
         RelativeErrorVariance=unname(rel), AbsoluteErrorVariance=unname(abs), RelativeSEM=unname(sqrt(rel)), AbsoluteSEM=unname(sqrt(abs)))
  })
  list(name=name, components=fit$components, diagnostics=fit$component_diagnostics,
       gram=fit$estimation$kernel_gram, rank=fit$estimation$rank, raw_plans=raw$coefficients, clipped_plans=clipped)
})
jsonlite::write_json(results, file.path(args[2L], 'unbalanced_gd_mfrmr.json'), digits=NA, auto_unbox=TRUE)
