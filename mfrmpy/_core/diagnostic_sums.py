"""Grouped diagnostic sufficient sums without constructing a frame per person."""
import numpy as np
import pandas as pd
from scipy.stats import chi2, t


def groups(data, facets):
    for facet in facets:
        labels = pd.Categorical(data[facet])
        codes, valid = labels.codes, labels.codes >= 0
        count = len(labels.categories)
        def total(values):
            values = np.asarray(values, dtype=float)
            return np.bincount(codes[valid], weights=np.where(np.isnan(values[valid]), 0., values[valid]), minlength=count)
        yield facet, labels.categories, total


def weights(data):
    w = data.Weight.to_numpy(dtype=float) if 'Weight' in data else np.ones(len(data))
    return np.where(np.isfinite(w) & (w > 0), w, 0.)


def facet_se(data, facets):
    w = weights(data)
    information = data.get('ScoreInformation', data.Var).to_numpy()*w
    frames = []
    for facet, labels, total in groups(data, facets):
        info = total(information)
        with np.errstate(divide='ignore', invalid='ignore'):
            se = np.where(info > 0, 1/np.sqrt(info), np.nan)
        frames.append(pd.DataFrame(dict(Facet=facet, Level=labels, N=total(w), SE=se)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def facet_bias(data, facets):
    w = weights(data)
    frames = []
    arrays = {key: data[key].to_numpy(dtype=float) for key in ('Observed','Expected','Residual','StdResidual','Var')}
    for facet, labels, total in groups(data, facets):
        n = total(w)
        def mean(values):
            ok = np.isfinite(values) & (w > 0)
            den = total(np.where(ok,w,0.))
            with np.errstate(divide='ignore', invalid='ignore'):
                return total(np.where(ok, values*w, 0.))/den
        observed, expected, residual, standard = (mean(arrays[k]) for k in ('Observed','Expected','Residual','StdResidual'))
        chi = total(arrays['StdResidual']**2*w)
        df = np.where(n > 1, n-1, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            se = np.sqrt(total(arrays['Var']*w))/n
            se_std = np.where(n > 0, 1/np.sqrt(n), np.nan)
            tr = np.where(np.isfinite(se) & (se > 0), residual/se, np.nan)
            ts = np.where(np.isfinite(se_std) & (se_std > 0), standard/se_std, np.nan)
        pr = np.where(np.isfinite(tr), 2*t.cdf(-np.abs(tr),df), np.nan)
        ps = np.where(np.isfinite(ts), 2*t.cdf(-np.abs(ts),df), np.nan)
        pc = np.where(np.isfinite(chi), 1-chi2.cdf(chi,df), np.nan)
        frames.append(pd.DataFrame(dict(Facet=facet, Level=labels, N=n, ObservedAverage=observed,
            ExpectedAverage=expected, Bias=residual, MeanResidual=residual, MeanStdResidual=standard,
            MeanAbsStdResidual=mean(np.abs(arrays['StdResidual'])), ChiSq=chi, ChiDf=df, ChiP=pc,
            SE_Residual=se, t_Residual=tr, p_Residual=pr, SE_StdResidual=se_std,
            t_StdResidual=ts, p_StdResidual=ps, DF=df)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def facet_fit_rows(data, facets, *, whexact=False, zstd_transform=None):
    from .engine import _facets_fit_df, zstd_from_mnsq
    n = len(data)
    def values(name):
        return pd.to_numeric(data[name], errors='coerce').to_numpy(dtype=float) if name in data else np.full(n,np.nan)
    w = weights(data)
    var, sq, fourth = (values(k) for k in ('Var','StdSq','FourthCentralMoment'))
    boundary = data.FitBoundaryProxy.fillna(False).to_numpy(dtype=bool) if 'FitBoundaryProxy' in data else np.zeros(n,dtype=bool)
    observed, expected = values('Observed'), values('Expected')
    zero = boundary & ~np.isfinite(var) & np.isfinite(observed) & np.isfinite(expected) & (np.abs(observed-expected) < 1e-4) & (w > 0)
    available = (np.isfinite(sq) & (sq >= 0) & np.isfinite(var) & (var > 0) & (w > 0)) | zero
    var_eff, sq_eff = np.where(zero,0.,var), np.where(zero,0.,sq)
    valid_df = np.isfinite(var) & (var > 1e-12) & np.isfinite(fourth) & (w > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        di = np.where(valid_df, w*(fourth-var**2), 0.)
        do = np.where(valid_df, w*(fourth/var**2-1), 0.)
    rows = []
    for facet, labels, total in groups(data, facets):
        counts, good, edges, mass = (total(v) for v in (np.ones(n),available,boundary,w))
        variance, weighted_sq, simple_sq = (total(v) for v in (var_eff*w,sq_eff*var_eff*w,sq_eff*w))
        den_i, den_o = total(di), total(do)
        for i, label in enumerate(labels):
            infit = outfit = dfi = dfo = fi = fo = np.nan
            if counts[i] > 0 and good[i] == counts[i]:
                dfi, dfo = variance[i], mass[i]
                if dfi > 1e-10:
                    infit = weighted_sq[i]/dfi
                    outfit = simple_sq[i]/dfo if dfo > 0 else np.nan
                fi, fo = _facets_fit_df(dfi,dfo,den_i[i],den_o[i])
            rows.append(dict(Facet=facet, Level=label, N=mass[i], Infit=infit, Outfit=outfit,
                InfitZSTD=zstd_from_mnsq(infit,dfi,whexact=whexact,zstd_transform=zstd_transform),
                OutfitZSTD=zstd_from_mnsq(outfit,dfo,whexact=whexact,zstd_transform=zstd_transform),
                DF_Infit=dfi, DF_Outfit=dfo, DF_Infit_FACETS=fi, DF_Outfit_FACETS=fo,
                FitResponseRows=int(counts[i]), FitAvailableRows=int(good[i]),
                FitUnavailableRows=int(counts[i]-good[i]), FitBoundaryRows=int(edges[i])))
    return rows
