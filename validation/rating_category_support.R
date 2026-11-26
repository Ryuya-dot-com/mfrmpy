# Use the same 1..5 score data for the R package probes.
path <- commandArgs(trailingOnly=TRUE)[1]
people <- c("A333","B124","CMIS")
variants <- c("base","rater_gap","global_gap")
cells <- expand.grid(Item=paste0("I",1:2),Rater=paste0("R",1:4))
ids <- with(cells,paste(Item,Rater))
items <- rbind(1,-1)[match(cells$Item,paste0("I",1:2)),,drop=FALSE]
raters <- rbind(diag(3),-1)[match(cells$Rater,paste0("R",1:4)),]
location <- cbind(items,raters)
steps <- rbind(diag(3),-1)
A <- array(0,c(8,5,7),dimnames=list(ids,as.character(0:4),paste0("p",1:7)))
for (k in 1:4) {
    A[,k+1,1:4] <- -k*location
    A[,k+1,5:7] <- -rep(colSums(steps[seq_len(k),,drop=FALSE]),each=8)
}
B <- array(rep(0:4,each=8),c(8,5,1))
W <- matrix(aperm(A[,-1,,drop=FALSE],c(2,1,3)),32,7)

capture <- function(task) {
    notes <- character()
    x <- tryCatch(withCallingHandlers(task(),warning=function(w) {
        notes <<- c(notes,conditionMessage(w)); invokeRestart("muffleWarning")
    }),error=function(e) list(status="error",detail=conditionMessage(e)))
    x$warnings <- paste(unique(notes),collapse=" | ")
    as.data.frame(x,stringsAsFactors=FALSE)
}

person_table <- function(engine, type, estimates, se, ids_arg, data) {
    i <- match(people,ids_arg)
    stopifnot(!anyNA(i), all(is.finite(estimates[i])), abs(estimates[i[1]]-estimates[i[2]])<1e-8)
    data.frame(engine=engine,type=type,person=people,ratings=as.integer(table(data$Person[!is.na(data$Score)])[people]),
               estimate=estimates[i], uncertainty=se[i])
}

all_cases <- list()
all_people <- list()
for (label in variants) {
    d <- read.csv(file.path(path,paste0(label,".csv")))
    persons <- sort(unique(d$Person))
    y <- matrix(NA_integer_,length(persons),8,dimnames=list(persons,ids))
    y[cbind(match(d$Person,persons),match(with(d,paste(Item,Rater)),ids))] <- d$Score-1L
    observed <- sum(!is.na(y))

    run <- function(engine, model, task) {
        case <- capture(task)
        all_cases[[length(all_cases)+1L]] <<- cbind(engine=engine,variant=label,model=model,case)
    }
    for (owner in if (label=="rater_gap") c("RSM","Rater") else "RSM") {
        for (policy in if (label=="global_gap") c("preserve","collapse") else "preserve") {
            run("mfrmr",paste(owner,policy),function() {
                fit <- mfrmr::fit_mfrm(d,"Person",c("Item","Rater"),"Score",rating_min=1,rating_max=5,
                    model=if (owner=="RSM") "RSM" else "PCM",step_facet=if (owner=="RSM") NULL else owner,
                    keep_original=policy=="preserve",category_policy=policy,quad_points=31,reltol=1e-10)
                stopifnot(nrow(fit$prep$data)==observed)
                if (label=="base") {
                    t <- fit$facets$person
                    all_people[[length(all_people)+1L]] <<- person_table("mfrmr","EAP",t$Estimate,t$SD,t$Person,d)
                }
                map <- fit$prep$score_map
                list(status="returned",detail=paste(paste(map$OriginalScore,map$InternalScore,sep="=>"),collapse=";"))
            })
        }
    }
    run("TAM","RSM: explicit A/B",function() {
        fit <- TAM::tam.mml(resp=y,A=A,B=B,variance.fixed=matrix(c(1,1,1),1),beta.fixed=matrix(c(1,1,0),1),
            control=list(nodes=seq(-8,8,length.out=161),conv=1e-10,convD=1e-10,convM=1e-10,maxiter=1000),verbose=FALSE)
        stopifnot(isTRUE(all.equal(unname(fit$resp),unname(y),check.attributes=FALSE)))
        if (label=="base") {
            t <- fit$person
            all_people[[length(all_people)+1L]] <<- person_table("TAM","EAP",t$EAP,t$SD.EAP,persons,d)
        }
        list(status=if(fit$iter<1000) "returned" else "iteration limit",
             detail=sprintf("categories=%d; iterations=%d; max.abs(step coordinates)=%.7f",fit$maxK,fit$iter,max(abs(fit$xsi$xsi[5:7]))))
    })
    run("eRm","LRSM: explicit W",function() {
        fit <- eRm::LRSM(y,W=.1*W,mpoints=1,groupvec=1,se=TRUE)
        stopifnot(isTRUE(all.equal(unname(fit$X),unname(y),check.attributes=FALSE)))
        if (label=="base") {
            t <- eRm::person.parameter(fit)$theta.table
            all_people[[length(all_people)+1L]] <<- person_table("eRm","conditional calibration / person ML",
                t[[1]],t[[2]],rownames(t),d)
        }
        list(status="returned",detail=paste("observed codes",paste(sort(unique(as.vector(fit$X))),collapse=","),
             "convergence",fit$convergence,"finite coefficient SEs",sum(is.finite(fit$se.eta)),"/",length(fit$se.eta)))
    })
    run("rasch","PCML: MFRM",function() {
        input <- transform(d,Score=Score-1L,Rubric="shared")
        fit <- rasch::rasch_mfrm(input,person="Person",item="Rubric",score="Score",facets=c("Item","Rater"))
        stopifnot(sum(!is.na(fit$X))==observed)
        if (label=="base") {
            t <- fit$person
            all_people[[length(all_people)+1L]] <<- person_table("rasch","PCML calibration / person WLE",
                t$theta,t$se,t$id,d)
            write.csv(t[match(people,t$id),c("id","infit_ms","outfit_ms")],file.path(path,"rasch_fit.csv"),row.names=FALSE)
        }
        list(status=if (isTRUE(fit$est$converged)) "returned" else "not converged",
             detail=paste("virtual maximum codes",paste(unname(fit$est$m),collapse=",")))
    })
}
write.csv(do.call(rbind,all_cases),file.path(path,"R_cases.csv"),row.names=FALSE)
write.csv(do.call(rbind,all_people),file.path(path,"R_people.csv"),row.names=FALSE)
