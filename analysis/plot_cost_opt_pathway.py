"""Cost-optimal pathway and the near-optimal space of each horizon, in two tech dimensions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
Y_TECH = "backup"

# Link p_nom is on the input side
LINK_OUTPUT_CAPACITY = True

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" if "__file__" in globals() else Path("../figures")
OUTPUT_PNG = OUTPUT_DIR / f"{X_TECH}_{Y_TECH}_cost_opt_pathway.png"  # set to None to skip saving

RESULTS = PYPSA_EUR / "results" / RUN
CACHE = PYPSA_EUR / "mga-cache"


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


def read_caps(caps_csv):
    caps = pd.read_csv(caps_csv)
    return caps[caps.attribute.isin(["p_nom_opt", "e_nom_opt"])].set_index(["component", "name"]).value


level_pts = {}
cost_opt_pts = {}
for h in HORIZONS:
    stem = f"base_s_{CLUSTERS}___{h}"
    info = component_info(pypsa.Network(RESULTS / "networks" / f"{stem}.nc"))
    network_hash = (RESULTS / "near_opt" / f"{stem}_network_hash.txt").read_text().strip()
    rows = {}
    for d in pd.read_csv(RESULTS / "near_opt" / f"{stem}.csv").dir_hash:
        caps_csv = CACHE / "caps" / f"caps_{network_hash}_{d}.csv"
        rows[d] = tech_capacities(read_caps(caps_csv), info, caps_csv.name)
    level_pts[h] = pd.DataFrame(rows).T[[X_TECH, Y_TECH]]
    caps_csv = RESULTS / "resilience" / f"cost_opt_caps_{stem}.csv"
    cost_opt_pts[h] = tech_capacities(read_caps(caps_csv), info, caps_csv.name)

for h, df in level_pts.items():
    print(f"{h}: {len(df)} directions, cost-optimal {X_TECH} {cost_opt_pts[h][X_TECH]/1e3:.2f} GW, "
          f"{Y_TECH} {cost_opt_pts[h][Y_TECH]/1e3:.2f} GW")
    print((df / 1e3).round(2).to_string(), "\n")


# plot
fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.get_cmap("viridis")
colors = [cmap(i / max(len(level_pts) - 1, 1)) for i in range(len(level_pts))]

for (h, df), color in zip(level_pts.items(), colors):
    pts_gw = df.values / 1e3
    try:
        hull = ConvexHull(pts_gw)
    except QhullError:
        print(f"{h}: points are degenerate (collinear/duplicate), no hull drawn")
        ax.scatter(*pts_gw.T, color=color, s=15, label=h)
    else:
        ring = pts_gw[list(hull.vertices) + [hull.vertices[0]]]
        ax.plot(*ring.T, color=color, alpha=0.8, linewidth=2, label=h)
        ax.fill(*ring.T, color=color, alpha=0.1, linewidth=0)

    co = cost_opt_pts[h]
    ax.scatter(co[X_TECH] / 1e3, co[Y_TECH] / 1e3, color=color, marker="*", s=200,
               edgecolor="black", linewidth=0.7, zorder=6)

# cost-optimal pathway: arrows from one horizon's cost-optimal network to the next
path = pd.DataFrame(cost_opt_pts).T / 1e3
for (_, a), (_, b) in zip(path.iloc[:-1].iterrows(), path.iloc[1:].iterrows()):
    ax.annotate("", xy=(b[X_TECH], b[Y_TECH]), xytext=(a[X_TECH], a[Y_TECH]),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2, shrinkA=8, shrinkB=8), zorder=5)

ax.scatter([], [], color="grey", marker="*", s=200, edgecolor="black", linewidth=0.7, label="cost-optimal")
ax.plot([], [], color="black", linewidth=1.2, label="cost-optimal pathway")

all_pts = pd.concat([*level_pts.values(), path * 1e3]) / 1e3
ax.set_xlim(0, 1.05 * all_pts[X_TECH].max())
ax.set_ylim(0, 1.05 * all_pts[Y_TECH].max())


def axis_label(tech):
    electric = TECHS[tech][0] == "Link" and LINK_OUTPUT_CAPACITY
    return f"{tech.capitalize()} capacity (GW{', electric output' if electric else ''})"


ax.set_xlabel(axis_label(X_TECH))
ax.set_ylabel(axis_label(Y_TECH))
ax.set_title(f"Near-optimal pathway in {X_TECH}-{Y_TECH} space")
ax.legend()
fig.tight_layout()

if OUTPUT_PNG is not None:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"saved {OUTPUT_PNG}")

plt.show()
