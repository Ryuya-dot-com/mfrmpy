# Check the first replicate at both person SD assumptions where applicable.
source("validation/m3_anchor_se_tam.R")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args)==4L, packageVersion("TAM")=="4.3.25")
ref <- read.csv(gzfile(args[1]))
source_data <- read.csv(args[2])
calibration <- split(source_data, source_data$profile)
ratings <- read.csv(args[3])
ref <- subset(ref, replicate==1L & profile %in% unique(ratings$profile) &
              design=="balanced24" & model %in% c("free", "estimated"))
rows <- list()
w <- c(0,0,0,-2,-1,0,0)
for (label in unique(ref$profile)) {
    src <- estimate(calibration[[label]], source_design)
    for (sd in unique(ref$assumed_person_sd[ref$profile==label])) {
        case <- subset(ref, profile==label & assumed_person_sd==sd)
        data <- subset(ratings, profile==label)
        free <- estimate(data, source_design, sd=sd)
        unanchored <- subset(case, model=="free")
        fixed <- subset(case, model=="estimated")
        stopifnot(nrow(unanchored)==1L, nrow(fixed)==1L)
        anchored <- estimate(data, target_design, fixed=matrix(c(5, fixed$anchor_R2-fixed$anchor_R1), 1), sd=sd)
        at <- match(6,anchored$free)
        derivative <- drop(-anchored$cov[at,] %*% anchored$h[anchored$free,5])
        free_se <- sqrt(drop(w %*% free$cov %*% w))
        fixed_se <- sqrt(anchored$cov[at,at])
        anchor_se <- abs(derivative)*sqrt(sum(src$cov[4:5,4:5]*matrix(c(1,-1,-1,1),2)))
        rows[[length(rows)+1L]] <- data.frame(profile=label, assumed_person_sd=sd,
            free_delta=sum(w*free$point)-unanchored$estimate, free_se_delta=free_se-unanchored$se,
            anchored_delta=anchored$point[6]-fixed$estimate, fixed_se_delta=fixed_se-fixed$se,
            propagated_se_delta=sqrt(fixed_se^2+anchor_se^2)-fixed$propagated_se,
            anchor_point_delta=max(abs(src$point[4:5]-c(fixed$anchor_R1,fixed$anchor_R2))))
    }
}
out <- do.call(rbind, rows)
stopifnot(nrow(out)>0L, all(abs(as.matrix(out[, 3:8]))<5e-5))
write.csv(out,args[4],row.names=FALSE)
