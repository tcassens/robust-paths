"""Per horizon: near-optimal space of every network an MGA ran on (cost-opt and pathway o networks),
every Chebyshev-centre network (c) of that horizon and the cost-optimal network, in two tech dimensions.

A c network is the re-solve pinned to the centre of its parent's MGA (in the MGA's EUR dimensions), so
it is drawn in the colour of its parent's near-optimal space. In these MW tech groups it can lie
slightly outside the 2D hull: the hull is a projection, and the pin only fixes the EUR totals per dim.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pypsa
from scipy.spatial import ConvexHull, QhullError

# config
PYPSA_EUR = Path("/work/users/s261224/sequential-mga/pypsa-eur")
RUN = "dk_24h/weather_year_2013_24H"
HORIZONS = ["2030", "2040", "2050"]
CLUSTERS = "2"

# tech groups
TECHS = {
    "wind": ("Generator", ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]),
    "solar": ("Generator", ["solar", "solar-hsat", "solar rooftop"]),
    "backup": ("Link", [
        "OCGT",
        "CCGT",
        "H2 Fuel Cell",
        "H2 turbine",
        "urban central gas CHP",
        "urban central gas CHP CC",
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
        "OCGT methanol",
    ]),
}

# select two techs
X_TECH = "solar"
Y_TECH = "wind"

# Link p_nom is on the input side
LINK_OUTPUT_CAPACITY = True

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" if "__file__" in globals() else Path("../figures")
SAVE = True  # one PNG per horizon: {X_TECH}_{Y_TECH}_near_opt_space_{horizon}.png

RESULTS = PYPSA_EUR / "results" / RUN
CACHE = PYPSA_EUR / "mga-cache"


def stem(h, pathway=""):
    """File stem, e.g. base_s_2___2040 (cost-opt) or base_s_2___2040_c2030-o2040."""
    return f"base_s_{CLUSTERS}___{h}" + (f"_{pathway}" if pathway else "")


def pathway_label(pathway):
    """Readable pathway, e.g. c2030-o2040 -> centre 2030 → cost-opt 2040; "" -> cost-optimal."""
    if not pathway:
        return "cost-optimal"
    steps = {"o": "cost-opt", "c": "centre"}
    return " → ".join(f"{steps[s[0]]} {s[1:]}" for s in pathway.split("-"))


def sort_key(pathway):
    """Order pathways by their o/c steps (o before c, earlier horizons first)."""
    return [s[0] == "c" for s in pathway.split("-")]


def mga_pathways(h):
    """Pathways whose network at horizon h has an MGA ("" = cost-opt network)."""
    base = stem(h)
    found = [f.stem[len(base) + 1:] for f in (RESULTS / "near_opt").glob(f"{base}_*.csv")]
    return [""] + sorted(found, key=sort_key)


def centre_pathways(h):
    """Pathways ending in a centre step at horizon h."""
    base = stem(h)
    found = [f.stem[len(base) + 1:] for f in (RESULTS / "networks").glob(f"{base}_*c{h}.nc")]
    return sorted(found, key=sort_key)


def parent_mga(pathway, h):
    """MGA a centre comes from: same pathway with the last step o instead of c ("" if that is all o)."""
    parent = pathway[: -len(f"c{h}")] + f"o{h}"
    return "" if all(step[0] == "o" for step in parent.split("-")) else parent


def component_info(n):
    """(component, name) -> carrier and capacity factor to plotted unit."""
    frames = []
    for c in {comp for comp, _ in TECHS.values()}:
        df = n.static(c)
        scale = df.efficiency if (c == "Link" and LINK_OUTPUT_CAPACITY) else pd.Series(1.0, df.index)
        frames.append(pd.DataFrame({"component": c, "name": df.index, "carrier": df.carrier, "scale": scale}))
    return pd.concat(frames).set_index(["component", "name"])


def tech_capacities(caps, info, source):
    """Sum capacity of X_TECH and Y_TECH from a (component, name) -> capacity series (MW)."""
    out = {}
    for tech in (X_TECH, Y_TECH):
        component, carriers = TECHS[tech]
        idx = info.index[(info.index.get_level_values(0) == component) & info.carrier.isin(carriers)]
        missing = idx.difference(caps.index)
        if len(missing):
            raise KeyError(f"{source}: {tech} components missing from caps: {list(missing)[:5]}")
        out[tech] = (caps.loc[idx] * info.loc[idx, "scale"]).sum()
    return out


def network_capacities(n, info, source):
    """X_TECH and Y_TECH capacities of a solved network (p_nom_opt / e_nom_opt)."""
    caps = []
    for c in {comp for comp, _ in TECHS.values()}:
        s = n.static(c)["e_nom_opt" if c == "Store" else "p_nom_opt"]
        caps.append(pd.Series(s.values, pd.MultiIndex.from_product([[c], s.index], names=["component", "name"])))
    return tech_capacities(pd.concat(caps), info, source)


def mga_points(h, pathway, info):
    """X_TECH and Y_TECH capacities of every MGA direction solved on the network (h, pathway)."""
    s = stem(h, pathway)
    network_hash = (RESULTS / "near_opt" / f"{s}_network_hash.txt").read_text().strip()
    rows = {}
    for d in pd.read_csv(RESULTS / "near_opt" / f"{s}.csv").dir_hash:
        caps_csv = CACHE / "caps" / f"caps_{network_hash}_{d}.csv"
        caps = pd.read_csv(caps_csv)
        caps = caps[caps.attribute.isin(["p_nom_opt", "e_nom_opt"])].set_index(["component", "name"]).value
        rows[d] = tech_capacities(caps, info, caps_csv.name)
    return pd.DataFrame(rows).T[[X_TECH, Y_TECH]]


def axis_label(tech):
    electric = TECHS[tech][0] == "Link" and LINK_OUTPUT_CAPACITY
    return f"{tech.capitalize()} capacity (GW{', electric output' if electric else ''})"


for h in HORIZONS:
    hulls, optima, centres = {}, {}, {}
    for p in mga_pathways(h):
        n = pypsa.Network(RESULTS / "networks" / f"{stem(h, p)}.nc")
        info = component_info(n)
        hulls[p] = mga_points(h, p, info)
        optima[p] = network_capacities(n, info, stem(h, p))
    for p in centre_pathways(h):
        n = pypsa.Network(RESULTS / "networks" / f"{stem(h, p)}.nc")
        centres[p] = network_capacities(n, component_info(n), stem(h, p))

    print(f"{h} ({X_TECH} / {Y_TECH}, GW)")
    for p, v in optima.items():
        print(f"  least-cost {v[X_TECH]/1e3:8.2f} {v[Y_TECH]/1e3:8.2f}  {pathway_label(p)} "
              f"({len(hulls[p])} MGA directions)")
    for p, v in centres.items():
        print(f"  centre     {v[X_TECH]/1e3:8.2f} {v[Y_TECH]/1e3:8.2f}  {pathway_label(p)} "
              f"(MGA on {pathway_label(parent_mga(p, h))})")

    # plot
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    colors = dict(zip(hulls, plt.get_cmap("tab10").colors))

    for p, df in hulls.items():
        pts_gw = df.values / 1e3
        try:
            hull = ConvexHull(pts_gw)
        except QhullError:
            print(f"{h} {pathway_label(p)}: points are degenerate (collinear/duplicate), no hull drawn")
            ax.scatter(*pts_gw.T, color=colors[p], s=15)
        else:
            ring = pts_gw[list(hull.vertices) + [hull.vertices[0]]]
            ax.plot(*ring.T, color=colors[p], alpha=0.8, linewidth=2)
            ax.fill(*ring.T, color=colors[p], alpha=0.1, linewidth=0)

        o = optima[p]
        ax.scatter(o[X_TECH] / 1e3, o[Y_TECH] / 1e3, color=colors[p], marker="*",
                   s=250 if p == "" else 120, edgecolor="black", linewidth=0.7, zorder=6)

    for p, c in centres.items():
        ax.scatter(c[X_TECH] / 1e3, c[Y_TECH] / 1e3, color=colors[parent_mga(p, h)], marker="X", s=90,
                   edgecolor="black", linewidth=0.7, zorder=7)

    # two legends outside the axes: colour = network the MGA ran on, marker = what is drawn.
    # Every such network ends in cost-opt h, so the colour legend lists only the steps before it.
    colour_handles = [
        Patch(facecolor=colors[p], edgecolor=colors[p], alpha=0.6,
              label=pathway_label("-".join(p.split("-")[:-1])) if p else "cost-optimal pathway")
        for p in hulls
    ]
    marker_kw = dict(color="none", markerfacecolor="grey", markeredgecolor="black", markeredgewidth=0.7)
    marker_handles = [
        Line2D([], [], color="grey", linewidth=2, label="near-optimal space"),
        Line2D([], [], marker="*", markersize=15, label="cost-optimal", **marker_kw),
    ]
    if len(optima) > 1:
        marker_handles.append(Line2D([], [], marker="*", markersize=11, label="least-cost solve", **marker_kw))
    marker_handles.append(Line2D([], [], marker="X", markersize=9, label="Chebyshev centre (re-solved)", **marker_kw))

    def header(text):
        return Line2D([], [], color="none", label=text)

    handles = [header(f"MGA on cost-opt {h} after:"), *colour_handles, header(" "), header("Markers:"), *marker_handles]
    fig.legend(handles=handles, fontsize=9, loc="outside right upper")

    all_pts = pd.concat([*hulls.values(), pd.DataFrame(optima).T, pd.DataFrame(centres).T]) / 1e3
    ax.set_xlim(0, 1.05 * all_pts[X_TECH].max())
    ax.set_ylim(0, 1.05 * all_pts[Y_TECH].max())
    ax.set_xlabel(axis_label(X_TECH))
    ax.set_ylabel(axis_label(Y_TECH))
    ax.set_title(f"{h}: near-optimal spaces and Chebyshev centres in {X_TECH}-{Y_TECH} space")

    if SAVE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{X_TECH}_{Y_TECH}_near_opt_space_{h}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {out}")

plt.show()
