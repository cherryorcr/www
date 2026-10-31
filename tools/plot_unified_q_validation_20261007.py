"""Plots separated by evidence scope, without selecting best cells."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "confirmation": (["uq_dm_edgefree","uq_dm_edgefree_no_gm","cat_cctp"],
                     ["CaT-Q","no GM","CaT-CCTP"], "9 order/construction cells; SD across cell means"),
    "baselines": (["uq_dm_edgefree","uq_dm_edgefree_no_gm","cat_cctp","gecc_hard","puma_tag","random_coreset","kcenter_coreset"],
                  ["CaT-Q","no GM","CaT-CCTP","GECC-hard","PUMA-inspired","Random","K-center"], "Reference order and construction seed; SD across 3 Student seeds"),
    "attribution": (["unchanged","text_blank","text_cctp","text_permuted","topology_hard","topology_weighted","topology_permuted"],
                    ["Self-loop","Blank text","CCTP text","Perm. text","Hard graph","Weighted","Shuffled graph"], "Fixed content/operator attribution, not reoptimized methods; 3 Student seeds"),
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    data=json.loads((args.root/"summary.json").read_text())
    output=args.root/"figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for kind,(arms,labels,subtitle) in GROUPS.items():
        rows=(data["aggregates"] if kind=="confirmation" else [r for r in data["cells"]
              if r["order_seed"]==9101 and r["construction_seed"]==271006])
        index={(r["dataset"],r["arm"]):r for r in rows}
        if not all((dataset,arm) in index for dataset in ("wikics","sportsfit","bookhis") for arm in arms):
            print(f"skip incomplete plot: {kind}"); continue
        fig,axes=plt.subplots(3,2,figsize=(13.2,10.5),layout="constrained")
        for row,dataset in enumerate(("wikics","sportsfit","bookhis")):
            for column,metric in enumerate(("final_f1","accuracy")):
                ax=axes[row,column]
                values=[index[(dataset,a)][metric] for a in arms]
                sd=("f1_cell_sd" if metric=="final_f1" else "acc_cell_sd") if kind=="confirmation" else (
                    "student_f1_sd" if metric=="final_f1" else "student_acc_sd")
                error=[index[(dataset,a)][sd] for a in arms]
                colors=["#237d72","#3c6898","#cf823d","#877795","#77894a","#9c6c70","#687982"][:len(arms)]
                bars=ax.bar(np.arange(len(arms)),values,yerr=error,capsize=3,color=colors,width=.66)
                ax.set_xticks(np.arange(len(arms)),labels,rotation=30 if len(arms)>3 else 0,ha="right" if len(arms)>3 else "center")
                ax.set_title(f"{dataset} | {'Macro-F1' if metric=='final_f1' else 'Accuracy'}")
                ax.set_ylim(0,100); ax.set_ylabel("Percent")
                ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
                for bar,value,err in zip(bars,values,error):
                    ax.text(bar.get_x()+bar.get_width()/2,value+err+1.2,f"{value:.1f}",ha="center",fontsize=8)
        fig.suptitle(f"In-loop CaT-Q: {kind}\n{subtitle}\nReport-dev only; public methods are explicit mechanism adaptations",fontsize=12)
        fig.savefig(output/f"{kind}.png",dpi=160)
        fig.savefig(output/f"{kind}.pdf")
        plt.close(fig)
        print(output/f"{kind}.png")


if __name__=="__main__":
    main()
