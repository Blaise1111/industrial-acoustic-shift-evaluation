"""Recompute post-hoc threshold and cost analyses without audio, weights or training."""
import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.cost_and_rules import policy_rows
from src.evidence_reaudit import (paired_class_draws, recalibrate, reviewed_outcome,
                                 routing_counts, split_source_proxy)
from src.pipeline import ROOT, WINDOWS, read_csv, write_csv
from src.reports import validate_ledger
from src.stats_and_calib import fewshot_rows, ranking
from sklearn.metrics import roc_auc_score


def sensitivity(group, original, config, identity):
    rows = []
    test = group[group.split.eq("target_test")]
    for p in policy_rows(group, original):
        if p["policy"] != "planned":
            continue
        counts = routing_counts(test.score, test.label, p["low"], p["high"])
        fields = {f"{label}_{action}_count": int(counts[i,j])
                  for i,label in enumerate(("normal","anomaly"))
                  for j,action in enumerate(("pass","review","alert"))}
        for pi, fn, quality in product(config["prevalences"], config["false_negative_costs"],
                                       config["review_quality"]):
            spec = config["review_quality"][quality]
            metrics = reviewed_outcome(counts, pi, fn, original["false_positive_cost"],
                                       original["review_cost"], **spec)
            if pi == original["prevalence"] and fn == p["fn_cost"] and quality == "perfect":
                for key, legacy in (("weighted_target_review_rate","review_rate"),
                                    ("raw_target_review_rate","observed_review_rate"),
                                    ("system_cost","cost"), ("system_recall","recall"),
                                    ("matched_random_cost","best_trivial_random_cost")):
                    np.testing.assert_allclose(metrics[key],p[legacy],atol=1e-7,rtol=0)
            rows.append(dict(**identity, **fields, planning_fn_cost=p["fn_cost"],
                             planning_prevalence=original["prevalence"],
                             planning_review_sensitivity=1., planning_review_specificity=1.,
                             planning_anomaly_rank_power=original["planning_anomaly_rank_power"],
                             budget=p["budget"], low=p["low"], high=p["high"],
                             source_planned_review_rate=p["planning_review_rate"],
                             evaluation_prevalence=pi, evaluation_fn_cost=fn, quality=quality,
                             review_cost=original["review_cost"],
                             false_positive_cost=original["false_positive_cost"], **spec, **metrics,
                             weighted_budget_exceeded=metrics["weighted_target_review_rate"]>p["budget"]+1e-12,
                             raw_budget_exceeded=metrics["raw_target_review_rate"]>p["budget"]+1e-12))
    return rows


def calibration(group, original, config, identity, rng):
    source = group[group.split.eq("source_calib")]
    proxy = group[group.split.eq("target_fewshot")]
    test = group[group.split.eq("target_test")]
    fixed = recalibrate(source.score.to_numpy(), proxy[WINDOWS].to_numpy(), original)
    membership, stability, effects, draws = [], [], [], {}
    partitions = [split_source_proxy(len(source),original["fewshot_clips"],
                                    config["source_fit_fraction"],rng)
                  for _ in range(config["source_proxy_replicates"])]
    for repeat, (fit, pseudo, evaluate) in enumerate(partitions):
        membership.append(dict(**identity, repeat=repeat, **{
            key:source.iloc[ids].clip_id.tolist()
            for key,ids in (("fit",fit),("proxy",pseudo),("evaluation",evaluate))}))
        scores = source.score.to_numpy()
        for method, threshold in recalibrate(scores[fit],source.iloc[pseudo][WINDOWS].to_numpy(),
                                            original).items():
            stability.append(dict(**identity, repeat=repeat, method=method, threshold=threshold,
                                  fit_n=len(fit),proxy_n=len(pseudo), evaluation_n=len(evaluate),
                                  held_out_fpr=np.mean(scores[evaluate]>threshold)))
    y = test.label.to_numpy()
    indices = paired_class_draws(y,config["bootstrap_replicates"],rng)
    scores = test.score.to_numpy()
    alerts = scores > fixed["Uncalibrated"]
    for method, threshold in fixed.items():
        shifted = scores > threshold
        delta = shifted.astype(float)-alerts
        bootstrap = np.stack([delta[samples].mean(axis=1) for samples in indices],axis=1)
        draws[method]=bootstrap
        lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
        operating = [float(shifted[y==c].mean()) for c in (0,1)]
        effects.append(dict(**identity, method=method, threshold=threshold,
                            method_kind="window_quantile_heuristic" if method=="Window q95" else "clip",
                            source_threshold=fixed["Uncalibrated"], target_calibration_n=len(proxy),
                            target_normal_n=len(indices[0][0]),target_anomaly_n=len(indices[1][0]),
                            fpr=operating[0], recall=operating[1], fpr_delta=delta[y==0].mean(),
                            recall_delta=delta[y==1].mean(),fpr_ci_low=lo[0],fpr_ci_high=hi[0],
                            recall_ci_low=lo[1],recall_ci_high=hi[1]))
    return effects, stability, membership, draws


def plot_cost(frame, output, focus):
    subset=frame[frame.budget==focus]
    groups=list(subset.groupby(["machine","planning_fn_cost"]))
    if not groups:
        raise ValueError("Focus budget is absent from the frozen plan")
    n_sections=subset.section.nunique()
    fig,axes=plt.subplots(len(groups),n_sections,figsize=(13.5,3.0*len(groups)),
                         layout="constrained",squeeze=False)
    limit=max(abs(subset.cost_saving_vs_random))
    for row,((machine,plan),m) in enumerate(groups):
        for col,(section,s) in enumerate(m.groupby("section")):
            ax=axes[row,col]
            groups=s.pivot(index=["evaluation_prevalence","evaluation_fn_cost"],
                           columns="quality",values="cost_saving_vs_random")
            ax.imshow(groups,cmap="RdBu",vmin=-limit,vmax=limit,aspect="auto")
            ax.set_title(f"{machine.title()} / {section} / plan {plan:g}:1")
            ax.set_xticks(range(len(groups.columns)),groups.columns,rotation=20,ha="right")
            ax.set_yticks(range(len(groups)),[f"{pi:.0%} / {cost:g}:1" for pi,cost in groups.index])
            for i,(pi,cost) in enumerate(groups.index):
                for j,quality in enumerate(groups.columns):
                    item=s[(s.evaluation_prevalence==pi)&(s.evaluation_fn_cost==cost)&(s.quality==quality)].iloc[0]
                    flags=("W" if item.weighted_budget_exceeded else "")+("R" if item.raw_budget_exceeded else "")
                    saving=groups.loc[(pi,cost),quality]
                    color="white" if abs(saving) > .55 * limit else "black"
                    ax.text(j,i,f"{saving:+.3f} {flags}".strip(),ha="center",va="center",fontsize=7,color=color)
    fig.suptitle(f"Cost saving vs matched random review at {focus:.0%} budget\n"
                 "Rows: assumed prevalence / evaluation cost ratio. W/R: weighted/raw budget exceeded.")
    fig.savefig(output,dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/evidence_reaudit.yaml")
    parser.add_argument("--ledgers",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    history=(ROOT/"results").resolve()
    if output==history or history in output.parents:
        parser.error("Use a dedicated output directory outside historical results")
    output.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    signatures={}
    all_cost, all_effect, all_stability, all_members, macro=[],[],[],[],[]
    historical_proxy,ranking_rows=[],[]
    for m in ("fan","valve"):
        path=args.ledgers/m/"predictions_clip_level.csv"
        ledger=read_csv(path)
        with path.open("rb") as stream:
            signatures[m]=file_digest(stream,"sha256").hexdigest()
        contract=json.loads(ledger.loc[ledger.analysis_config.ne(""),"analysis_config"].item())
        validate_ledger(ledger,contract)
        original=contract["evaluation"]
        ensemble=ledger[(ledger.model=="M2")&(ledger.seed=="ensemble")]
        by_method={}
        machine_effect=[]
        for section, group in ensemble.groupby("section",sort=True):
            identity=dict(machine=m,section=section,model="M2 ensemble")
            all_cost.extend(sensitivity(group,original,config,identity))
            for sample in fewshot_rows(group,original,WINDOWS):
                if sample["kind"]=="simulation_summary":
                    historical_proxy.append(dict(**identity,diagnostic="fit_in",**sample))
            for domain in ("source","target"):
                test=group[group.split.eq(domain+"_test")]
                cap=original["max_fpr"]
                measures=ranking(test.label,test.score,cap)
                measures["pauc_area_over_cap"]=measures.pop("pauc")
                ranking_rows.append(dict(**identity,domain=domain,n_clips=len(test),max_fpr=cap,
                                random_pauc_area_over_cap=cap/2,random_pauc_mcclish=.5,
                                pauc_mcclish=roc_auc_score(test.label,test.score,max_fpr=cap),
                                **measures))
            rng=np.random.default_rng(np.random.SeedSequence([config["seed"],int(m=="valve"),int(section)]))
            effects,stats,membership,draws=calibration(group.sort_values("clip_id"),original,config,identity,rng)
            machine_effect.extend(effects)
            all_stability.extend(stats)
            all_members.extend(membership)
            for method,values in draws.items():
                by_method.setdefault(method,[]).append(values)
        all_effect.extend(machine_effect)
        table=pd.DataFrame(machine_effect)
        for method,draws in by_method.items():
            lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975],axis=0)
            matched=table[table.method.eq(method)]
            macro.append(dict(machine=m,method=method,sections=len(draws),
                              **matched[["fpr","recall","fpr_delta","recall_delta"]].mean().to_dict(),
                              fpr_ci_low=lo[0], fpr_ci_high=hi[0], recall_ci_low=lo[1], recall_ci_high=hi[1]))
    cost=pd.DataFrame(all_cost)
    stability=pd.DataFrame(all_stability)
    grouping=["machine","planning_fn_cost","budget","evaluation_prevalence","evaluation_fn_cost","quality"]
    summary=cost.groupby(grouping).agg(
        sections=("section","size"), cost_saving_mean=("cost_saving_vs_random","mean"),
        sections_with_savings=("cost_saving_vs_random",lambda c:int((c>1e-12).sum())),
        weighted_budget_breaches=("weighted_budget_exceeded","sum"),
        raw_budget_breaches=("raw_budget_exceeded","sum"),
        model_recall_mean=("model_recall","mean"),system_recall_mean=("system_recall","mean"),
        weighted_target_review_mean=("weighted_target_review_rate","mean"),
        raw_target_review_mean=("raw_target_review_rate","mean")).reset_index()
    for name, frame in (("review_scenarios",cost),("review_macro",summary),
                        ("threshold_effects",pd.DataFrame(all_effect)),("threshold_macro",pd.DataFrame(macro)),
                        ("source_proxy_legacy_fit_in",pd.DataFrame(historical_proxy)),
                        ("pauc_definitions",pd.DataFrame(ranking_rows)),
                        ("source_proxy_trials",stability),("source_proxy_summary",stability.groupby(
                            ["machine","section","method"]).held_out_fpr.agg(["size","mean","std","min","max"]).reset_index())):
        write_csv(frame,output/f"{name}.csv")
    with (output/"source_proxy_partitions.jsonl").open("w",encoding="utf-8") as stream:
        for entry in all_members:
            stream.write(json.dumps(entry,separators=(",",":"))+"\n")
    (output/"analysis_manifest.json").write_text(json.dumps(dict(
        analysis="post_hoc_offline",baseline="baf1a8f",ledger_sha256=signatures,
        parameters=config,scope="M2 ensemble q90; frozen source-only planning"),indent=2),encoding="utf-8")
    plot_cost(cost,output/"cost_budget_sensitivity.png",config["focus_budget"])
    print(f"Recomputed {len(cost)} review scenario rows and {len(all_effect)} fixed-threshold comparisons")


if __name__=="__main__":
    main()
