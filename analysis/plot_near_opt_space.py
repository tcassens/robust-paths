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

WIND_CARRIERS = ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
SOLAR_CARRIERS = ["solar", "solar-hsat", "solar rooftop"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" if "__file__" in globals() else Path("../figures")
OUTPUT_PNG = OUTPUT_DIR / "wind_solar_near_opt_space_per_horizon.png"  # set to None to skip saving

RESULTS = PYPSA_EUR / "results" / RUN
CACHE = PYPSA_EUR / "mga-cache"


def carrier_map(horizon):
    """(component, name) -> carrier for every Generator in the horizon's solved network."""
    n = pypsa.Network(RESULTS / "networks" / f"base_s_{CLUSTERS}___{horizon}.nc")
    return n.generators.carrier


def wind_solar(caps_csv, carriers):
    """Sum p_nom_opt of wind and solar generators in a caps CSV (MW)."""
    caps = pd.read_csv(caps_csv)
    gens = caps[(caps.component == "Generator") & (caps.attribute == "p_nom_opt")]
    carrier = gens.name.map(carriers)
    if carrier.isna().any():
        missing = gens.name[carrier.isna()].tolist()
        raise KeyError(f"{caps_csv.name}: generators not in network: {missing[:5]}")
    return {
        "wind": gens.value[carrier.isin(WIND_CARRIERS)].sum(),
        "solar": gens.value[carrier.isin(SOLAR_CARRIERS)].sum(),
    }


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
    carriers = carrier_map(h)
    cand = pd.read_csv(RESULTS / "resilience" / f"mga_candidates_base_s_{CLUSTERS}___{h}.csv")
    rows = {}
    for r in cand.itertuples():
        direction = f"s{r.v_solar:+.2f} on{r.v_onwind:+.2f} off{r.v_offwind:+.2f} b{r.v_backup:+.2f}"
        rows[direction] = wind_solar(CACHE / "caps" / f"caps_{r.network_hash}_{r.direction_hash}.csv", carriers)
    level_pts[h] = pd.DataFrame(rows).T
    cost_opt_pts[h] = wind_solar(RESULTS / "resilience" / f"cost_opt_caps_base_s_{CLUSTERS}___{h}.csv", carriers)

for h, df in level_pts.items():
    print(f"{h}: {len(df)} directions, cost-optimal wind {cost_opt_pts[h]['wind']/1e3:.2f} GW, "
          f"solar {cost_opt_pts[h]['solar']/1e3:.2f} GW")
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
    ax.scatter(co["wind"] / 1e3, co["solar"] / 1e3, color=color, marker="*", s=200,
               edgecolor="black", linewidth=0.7, zorder=6)

# proxy legend entries for the marker meanings
ax.scatter([], [], color="grey", marker="*", s=200, edgecolor="black", linewidth=0.7, label="cost-optimal")
ax.scatter([], [], color="grey", marker="X", s=40, edgecolor="black", linewidth=0.5, label="Chebyshev center")

upper = 1.05 * max(df.values.max() for df in level_pts.values()) / 1e3
ax.set_xlim(0, upper)
ax.set_ylim(0, upper)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Wind capacity (GW)")
ax.set_ylabel("Solar capacity (GW)")
ax.set_title("Near-optimal space in wind-solar coordinate space")
ax.legend()
fig.tight_layout()

if OUTPUT_PNG is not None:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"saved {OUTPUT_PNG}")

plt.show()
