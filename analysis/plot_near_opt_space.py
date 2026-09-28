from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from scipy.optimize import linprog
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
OUTPUT_PNG = OUTPUT_DIR / f"{X_TECH}_{Y_TECH}_near_opt_space_per_horizon.png"  # set to None to skip saving

RESULTS = PYPSA_EUR / "results" / RUN
CACHE = PYPSA_EUR / "mga-cache"


def component_info(horizon):
    """(component, name) -> carrier and capacity factor to plotted unit, from the horizon's solved network."""
    n = pypsa.Network(RESULTS / "networks" / f"base_s_{CLUSTERS}___{horizon}.nc")
    frames = []
    for c in {comp for comp, _ in TECHS.values()}:
        df = n.static(c)
        scale = df.efficiency if (c == "Link" and LINK_OUTPUT_CAPACITY) else pd.Series(1.0, df.index)
        frames.append(pd.DataFrame({"component": c, "name": df.index, "carrier": df.carrier, "scale": scale}))
    return pd.concat(frames).set_index(["component", "name"])


def tech_capacities(caps_csv, info):
    """Sum capacity of X_TECH and Y_TECH in a caps CSV (MW)."""
    caps = pd.read_csv(caps_csv)
    caps = caps[caps.attribute.isin(["p_nom_opt", "e_nom_opt"])].set_index(["component", "name"]).value
    out = {}
    for tech in (X_TECH, Y_TECH):
        component, carriers = TECHS[tech]
        idx = info.index[(info.index.get_level_values(0) == component) & info.carrier.isin(carriers)]
        missing = idx.difference(caps.index)
        if len(missing):
            raise KeyError(f"{caps_csv.name}: {tech} components missing from caps: {list(missing)[:5]}")
        out[tech] = (caps.loc[idx] * info.loc[idx, "scale"]).sum()
    return out


def chebyshev_center(points):
    """Point maximizing the minimum distance to every edge of the convex hull of `points`."""
    hull = ConvexHull(points)
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    # maximize r s.t. A_i . x + r <= -b_i for every facet (hull.equations is unit-normalized)
    c = np.array([0, 0, -1])
    A_ub = np.hstack([A, np.ones((A.shape[0], 1))])
    b_ub = -b
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)])
    return res.x[:2], res.x[2]  # center, margin (distance to nearest edge)


level_pts = {}
cost_opt_pts = {}
for h in HORIZONS:
    info = component_info(h)
    cand = pd.read_csv(RESULTS / "resilience" / f"mga_candidates_base_s_{CLUSTERS}___{h}.csv")
    rows = {}
    for r in cand.itertuples():
        direction = f"s{r.v_solar:+.2f} on{r.v_onwind:+.2f} off{r.v_offwind:+.2f} b{r.v_backup:+.2f}"
        rows[direction] = tech_capacities(CACHE / "caps" / f"caps_{r.network_hash}_{r.direction_hash}.csv", info)
    level_pts[h] = pd.DataFrame(rows).T[[X_TECH, Y_TECH]]
    cost_opt_pts[h] = tech_capacities(RESULTS / "resilience" / f"cost_opt_caps_base_s_{CLUSTERS}___{h}.csv", info)

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
        for k, simplex in enumerate(hull.simplices):
            ax.plot(
                pts_gw[simplex, 0],
                pts_gw[simplex, 1],
                color=color,
                alpha=0.8,
                linewidth=2,
                label=h if k == 0 else None,
            )
        center, margin = chebyshev_center(pts_gw)
        ax.scatter(*center, color=color, marker="X", s=40, edgecolor="black", linewidth=0.5, zorder=5)
        print(f"{h}: Chebyshev center = ({center[0]:.2f}, {center[1]:.2f}) GW, margin = {margin:.3f} GW, "
              f"hull area = {hull.volume:.1f} GW²")

    co = cost_opt_pts[h]
    ax.scatter(co[X_TECH] / 1e3, co[Y_TECH] / 1e3, color=color, marker="*", s=200,
               edgecolor="black", linewidth=0.7, zorder=6)

ax.scatter([], [], color="grey", marker="*", s=200, edgecolor="black", linewidth=0.7, label="cost-optimal")
ax.scatter([], [], color="grey", marker="X", s=40, edgecolor="black", linewidth=0.5, label="Chebyshev center")

all_pts = pd.concat(level_pts.values()) / 1e3
ax.set_xlim(0, 1.05 * all_pts[X_TECH].max())
ax.set_ylim(0, 1.05 * all_pts[Y_TECH].max())
def axis_label(tech):
    electric = TECHS[tech][0] == "Link" and LINK_OUTPUT_CAPACITY
    return f"{tech.capitalize()} capacity (GW{', electric output' if electric else ''})"


ax.set_xlabel(axis_label(X_TECH))
ax.set_ylabel(axis_label(Y_TECH))
ax.set_title(f"Near-optimal space in {X_TECH}-{Y_TECH} coordinate space")
ax.legend()
fig.tight_layout()

if OUTPUT_PNG is not None:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"saved {OUTPUT_PNG}")

plt.show()
