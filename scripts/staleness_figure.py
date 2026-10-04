import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", help="one staleness_probe.py json per backbone")
ap.add_argument("--panels", default="a,b")
ap.add_argument("--out", default="")
ap.add_argument("--width", type=float, default=0.0,
                help="inches; 0 uses the ICLR text width (5.5 in) so the figure "
                     "goes in at width=\\textwidth with no rescaling, which is "
                     "what keeps its type the same size as the caption's")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--title", default="")
args = ap.parse_args()

PANELS = [p.strip() for p in args.panels.split(",") if p.strip()]
NAMES = {"pi0": r"$\pi_0$ (LIBERO, 50-action chunk)",
         "pi05": r"$\pi_{0.5}$ (LIBERO, 10-action chunk)",
         "dboft": "DB-OFT (Bridge, 16-action chunk)"}
COL = {"identity": "#444444", "cold": "#d95f02", "warm": "#1b6ca8"}
MARK = {0.3: "v", 0.5: "o", 0.7: "^"}


def load(path):
    d = json.load(open(path))
    A, P, m = d["aggregate"], d["per_position"], d["meta"]

    def med(key):
        v = A.get(key)
        return statistics.median(v) if v else None

    return dict(meta=m, med=med, pos=P,
                label=NAMES.get(m["model"], m["model"]))


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": args.dpi})

RUNS = [load(f) for f in args.files]
NR, NP = len(RUNS), len(PANELS)
W = args.width or (5.5 if NP == 2 else 2.75 * NP)
LEG_ROWS = -(-(4 + len(RUNS[0]["meta"]["taus"])) // 3)
LEG_H = 0.155 * LEG_ROWS + 0.02
fig, axes = plt.subplots(NR, NP, figsize=(W, 0.40 * W * NR + LEG_H),
                         squeeze=False)


def variants(meta):
    return ["identity", "cold"] + [f"warm{t:g}" for t in meta["taus"]]


def style(tag, meta):
    if tag == "identity":
        return dict(color=COL["identity"], marker="s", ls="--", ms=3.5,
                    label="do nothing (stale chunk)")
    if tag == "cold":
        return dict(color=COL["cold"], marker="d", ls=":", ms=3.5,
                    label="one step from noise")
    t = float(tag[4:])
    main = abs(t - meta["tau_main"]) < 1e-9
    return dict(color=COL["warm"], marker=MARK.get(t, "o"),
                ls="-" if main else "-.", ms=4 if main else 3,
                lw=1.8 if main else 1.0,
                alpha=1.0 if main else 0.55,
                label=rf"one step from the chunk, $\tau_0$={t:g}"
                      + (" (ours)" if main else ""))


def panel_a(ax, run, first, top):
    meta, med = run["meta"], run["med"]
    m = meta["m_main"]
    ks = [k for k in meta["probes"] if med(f"k{k}/m{m}/ref/dnn")]
    nn = np.array([med(f"k{k}/m{m}/ref/dnn") for k in ks])
    sp = np.array([med(f"k{k}/m{m}/ref/spread") for k in ks])
    ax.fill_between(ks, 0, nn, color="#1b6ca8", alpha=0.15, lw=0,
                    label="inside: the nearest-plan band")
    ax.fill_between(ks, nn, sp, color="#1b6ca8", alpha=0.07, lw=0,
                    label="within the typical plan-to-plan spread")
    ax.plot(ks, nn, color="#1b6ca8", lw=0.7, alpha=0.5)
    ax.plot(ks, sp, color="#1b6ca8", lw=0.7, alpha=0.35)
    hi = max(sp.max(), max(med(f"k{k}/m{m}/identity/dnn") for k in ks))
    for tag in variants(meta):
        ax.plot(ks, [med(f"k{k}/m{m}/{tag}/dnn") for k in ks], **style(tag, meta))
    ax.set_xlabel("actions executed since the plan, $k$")
    ax.set_ylabel(f"$d_{{nn}}$ over the next {m} actions")
    ax.set_ylim(0, hi * 1.12)
    ax.set_xlim(0, max(ks) + 1)
    r = [med(f"k{k}/m{m}/identity/dnn") / med(f"k{k}/m{m}/ref/spread") for k in ks]
    i = next((i for i, v in enumerate(r) if v > 1.0), None)
    if i:
        k0, k1, r0, r1 = ks[i - 1], ks[i], r[i - 1], r[i]
        cross = k0 + (1.0 - r0) * (k1 - k0) / (r1 - r0)
        ax.axvline(cross, color="#444444", ls=":", lw=0.9)
        ax.annotate(rf"$k^{{*}}\!\approx\!{cross:.0f}$", xy=(cross, hi * 1.06),
                    fontsize=7, ha="center", va="top", color="#444444",
                    bbox=dict(fc="white", ec="none", pad=1))
        print(f"  {run['label']}: k* = {cross:.1f} (band: typical spread, m={m})")
    ax.set_title("does the chunk go stale?" if top else "")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.98, 0.04, run["label"], transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7)


def panel_b(ax, run, first, top):
    meta, med = run["meta"], run["med"]
    m = meta["m_main"]
    ks = [k for k in meta["probes"] if med(f"k{k}/m{m}/ref/dnn")]
    ax.add_patch(plt.Rectangle((0, 0.85), 1.0, 0.3, color="#1b6ca8",
                               alpha=0.12, lw=0))
    ax.axvline(1.0, color="#1b6ca8", lw=0.7, alpha=0.5)
    ax.axhline(1.0, color="#1b6ca8", lw=0.7, alpha=0.5)
    for tag in variants(meta):
        st = style(tag, meta)
        x = [med(f"k{k}/m{m}/{tag}/dnn") / med(f"k{k}/m{m}/ref/dnn") for k in ks]
        y = [med(f"k{k}/m{m}/{tag}/dcent") / med(f"k{k}/m{m}/ref/dcent") for k in ks]
        ax.plot(x, y, **{**st, "ls": "-", "lw": st.get("lw", 1.0) * 0.7})
        ax.scatter(x[-1:], y[-1:], s=26, facecolor="none",
                   edgecolor=st["color"], zorder=5, lw=0.9)
    ax.set_xlabel(r"$d_{nn}$ / nearest-plan band  $\rightarrow$ stale")
    ax.set_ylabel(r"$d_{cent}$ / reference    $\downarrow$ averaged")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.annotate("the conditional mean:\na chunk the policy never samples",
                xy=(0.04, 0.05), xycoords="axes fraction", fontsize=6,
                color=COL["cold"])
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("is the repaired chunk a real plan?" if top else "")


def panel_c(ax, run, first, top):
    meta, pos = run["meta"], run["pos"]
    tag = f"warm{meta['tau_main']:g}"
    ks = [k for k in meta["probes"] if f"k{k}/ref/dnn_agg" in pos]
    if not ks:
        ax.text(0.5, 0.5, "per-position data needs a run with dnn_agg",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        return None
    H = meta["H"]

    def surface(tag):
        g = np.full((len(ks), H), np.nan)
        for i, k in enumerate(ks):
            ref = np.array(pos[f"k{k}/ref/dnn_agg"])
            cur = np.array(pos[f"k{k}/{tag}/dnn_agg"])
            n = min(len(ref), len(cur))
            r = cur[:n] / np.maximum(ref[:n], 1e-6)
            if n >= 3:
                r = np.convolve(r, np.ones(3) / 3, mode="same")
                r[0] = r[:2].mean()
                r[-1] = r[-2:].mean()
            g[i, :n] = r
        return g

    stale, fixed = surface("identity"), surface(tag)
    im = ax.imshow(stale, aspect="auto", origin="lower", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=0, vcenter=1.0,
                                     vmax=float(np.nanpercentile(stale, 98))),
                   extent=[0, H, -0.5, len(ks) - 0.5], interpolation="nearest")
    X, Y = np.meshgrid(np.arange(H) + 0.5, np.arange(len(ks)))
    for g, ls, lab in ((stale, "-", "stale: still inside"),
                       (fixed, "--", "after one correction")):
        ax.contour(X, Y, np.nan_to_num(g, nan=99.0), levels=[1.0],
                   colors="black", linewidths=1.2, linestyles=ls)
        ax.plot([], [], color="black", ls=ls, lw=1.2, label=lab)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks], fontsize=6)
    ax.set_xlabel("position in the chunk, $h$")
    ax.set_ylabel("actions since the plan, $k$")
    ax.set_xlim(0, H)
    ax.legend(loc="upper right", fontsize=5.8, framealpha=0.85)
    ax.set_title("where in the chunk the staleness sits" if top else "")
    return im


DRAW = {"a": panel_a, "b": panel_b, "c": panel_c}
ims = []
for r, run in enumerate(RUNS):
    for c, p in enumerate(PANELS):
        out = DRAW[p](axes[r][c], run, c == 0, r == 0)
        if out is not None:
            ims.append((out, axes[r][c]))

for im, ax in ims:
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05).ax.tick_params(labelsize=6)

h, l = axes[0][0].get_legend_handles_labels()
seen, hh, ll = set(), [], []
for a, b in zip(h, l):
    if b not in seen:
        seen.add(b)
        hh.append(a)
        ll.append(b)
if args.title:
    fig.suptitle(args.title, fontsize=9)
pad = LEG_H / (0.40 * W * NR + LEG_H)
fig.tight_layout(rect=(0, pad, 1, 0.97 if args.title else 1))
fig.legend(hh, ll, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.0), borderaxespad=0.3,
           columnspacing=1.2, handlelength=2.0)

base = args.out or os.path.join(os.path.dirname(args.files[0]) or ".",
                                "staleness_" + "".join(PANELS))
for ext in ("pdf", "png"):
    fig.savefig(f"{base}.{ext}", dpi=args.dpi)
    print(f"wrote {base}.{ext}  ({W:.2f} x {0.40 * W * NR + LEG_H:.2f} in)")
