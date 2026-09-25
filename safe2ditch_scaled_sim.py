"""Safe2Ditch-style emergency landing simulation.

- 1 m x 1 m cells on a 1 km x 1 km map.
- 10 randomized suburban map designs by default.
- 100 Monte Carlo failures per map before regenerating the map.
- Residential blocks use two back-to-back rows of 20 m x 30 m lots.
- Each lot contains one ~10 m x 15 m house; rear yards meet at the block center.
- Four parks and two schools occupy road-accessible road-bounded blocks, so their shapes
  follow the street layout rather than being forced to 100 m x 100 m squares.
- Three open/yard swaths span 2-4 former housing blocks each; their internal
  cross streets are omitted while the streets around them stay connected.
- Four parks occupy separate quadrants; the two schools are widely separated.
- Roads are 10 m or 15 m wide.
- People are represented by a center point plus a 5 m x 5 m danger zone.
- People may be centered on:
    * open cells directly adjacent to roads,
    * road cells directly adjacent to open land,
    * road cells adjacent to schools and school cells adjacent to roads,
    * any park cell.
    * a small, separately sampled fraction of residential backyards,
      with up to two people per backyard.
- Person centers in parks are capped at 7% of all people per trial,
  distributed across parks with a per-park ceiling.
- Compares baseline, Safe2Ditch without verification, and Safe2Ditch with verification.
- Saves every generated static map as an image and writes CSV summaries.
- Optional --people-map mode saves a selected design with one trial's person
  centers and 5 x 5 m danger zones overlaid, without running the experiment.
"""

from __future__ import annotations

import argparse
import csv
import math
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Simulation configuration
# -----------------------------------------------------------------------------

CELL_SIZE_M = 1
N = 1000                         # 1000 x 1000 cells -> 1 km x 1 km
NUM_MAPS = 10
TRIALS_PER_MAP = 100

MAP_MASTER_SEED = 12345
TRIAL_MASTER_SEED = 42

# Integer cell-type codes. Keep geometry/type separate from numerical risk.
OPEN, PARK, ROAD, HOUSE, SCHOOL, PERSON = range(6)
CELL_TYPES = (OPEN, PARK, ROAD, HOUSE, SCHOOL, PERSON)
STATIC_CELL_TYPES = (OPEN, PARK, ROAD, HOUSE, SCHOOL)
CELL_NAMES = {
    OPEN: "open",
    PARK: "park",
    ROAD: "road",
    HOUSE: "house",
    SCHOOL: "school",
    PERSON: "person",
}

# Person must be the highest-risk category.
RISK = np.array([
    0.05,   # OPEN
    0.20,   # PARK
    0.70,   # ROAD
    0.90,   # HOUSE
    0.95,   # SCHOOL
    1.00,   # PERSON / 5x5 m person danger zone
], dtype=float)

HIGH = 0.60                     # >= HIGH counts as a high-risk landing

# Map geometry, in meters. Because CELL_SIZE_M = 1, these equal cell counts.
ROAD_WIDTHS_M = (10, 15)
PLOT_DIMS_M = (20, 30)
HOUSE_DIMS_M = (10, 15)

# A suburban residential block is 60 m deep: one 30 m-deep row of lots
# fronts each of two opposite roads, so the rear lot lines meet in the middle.
RESIDENTIAL_BLOCK_DEPTH_M = 60
BLOCK_LENGTH_OPTIONS_M = (100, 120, 140, 160, 180)
FRONT_SETBACK_M = 5

# Parks/schools use whole road-bounded blocks when possible. These area bounds
# keep them roughly comparable to the old ~100x100 m target while allowing
# non-square shapes such as 60x140, 60x160, or 60x180 m.
SPECIAL_MIN_AREA_M2 = 8_000
SPECIAL_MAX_AREA_M2 = 12_000
NUM_PARKS = 4
NUM_SCHOOLS = 2
PARK_MIN_SEPARATION_M = 300
SCHOOL_MIN_SEPARATION_M = 500

# Each swath replaces a run of neighboring residential blocks and the short
# cross-street segments between them. Main frontage roads stay in place.
NUM_UNDEVELOPED_SWATHS = 3
UNDEVELOPED_BLOCK_COUNT_OPTIONS = (2, 3, 4)

# People / dynamic hazards.
# The old model used 60 whole 25x25 m cells as person hazards:
# 60 * 625 m^2 = 37,500 m^2 = 3.75% of a 1 km^2 map.
# With 5x5 m zones (25 m^2 each), 1500 centers gives the same gross area
# before overlap. Actual coverage is somewhat lower because zones can overlap.
PEOPLE_PER_TRIAL = 1500
PERSON_DANGER_SIZE_M = 5        # must be odd so there is one center cell
BACKYARD_PERSON_FRACTION = 0.13  # expected share of all people
MAX_PEOPLE_PER_BACKYARD = 2
PARK_PERSON_FRACTION = 0.07      # sampling target AND hard share ceiling

# Failure / reachability model.
MIN_RANGE_M = 25
MAX_RANGE_M = 150

# Old code used 0.01 per 25 m cell, which is 0.0004 per meter.
DISTANCE_WEIGHT_PER_M = 0.0004

# Site verification: initial site + up to 4 re-selections.
VERIFY_CANDIDATES = 5
# At 1 m resolution, do not treat immediately adjacent cells as distinct sites.
RESELECT_MIN_SEPARATION_M = 5
# We only need a small pool of top-ranked cells to find 5 distinct candidates.
CANDIDATE_POOL = 800

OUTPUT_DIR = Path("simulation_output_suburban_final")
MAP_DIR = OUTPUT_DIR / "maps"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_PATH = OUTPUT_DIR / "simulation_report.txt"
SHOW_MAP_IMAGES = False         # True = also display each map interactively

STRATEGIES = ("baseline", "s2d_noverify", "s2d")
STRATEGY_LABELS = {
    "baseline": "Drop in place",
    "s2d_noverify": "Map only",
    "s2d": "Map + verification",
}


# -----------------------------------------------------------------------------
# Map generation
# -----------------------------------------------------------------------------

def make_road_bands(
    rng: np.random.Generator,
    block_spans_m,
    n: int = N,
):
    """Create parallel road strips separated by specified non-road block spans."""
    bands = []
    pos = 0
    spans = tuple(int(x) for x in block_spans_m)

    while pos < n:
        width = int(rng.choice(ROAD_WIDTHS_M))
        end = min(pos + width, n)
        bands.append((pos, end))
        if end >= n:
            break

        block_span = int(rng.choice(spans))
        pos = end + block_span

    return bands


def free_intervals(bands, n: int = N):
    """Return non-road intervals between road bands."""
    intervals = []
    previous_end = 0

    for start, end in bands:
        if start > previous_end:
            intervals.append((previous_end, start))
        previous_end = max(previous_end, end)

    if previous_end < n:
        intervals.append((previous_end, n))

    return intervals


def road_sides_for_region(m: np.ndarray, r0: int, r1: int, c0: int, c1: int):
    """Return sides whose complete outside edge directly touches a road."""
    sides = []

    if r0 > 0 and np.all(m[r0 - 1, c0:c1] == ROAD):
        sides.append("top")
    if r1 < N and np.all(m[r1, c0:c1] == ROAD):
        sides.append("bottom")
    if c0 > 0 and np.all(m[r0:r1, c0 - 1] == ROAD):
        sides.append("left")
    if c1 < N and np.all(m[r0:r1, c1] == ROAD):
        sides.append("right")

    return tuple(sides)


def valid_plot_road_access(road_sides):
    """A residential lot needs 1 road side, or 2 adjacent road sides."""
    if len(road_sides) == 1:
        return True
    if len(road_sides) != 2:
        return False

    sides = set(road_sides)
    # Opposite road pairs would put streets at both the front and back of a lot.
    if sides == {"top", "bottom"} or sides == {"left", "right"}:
        return False
    return True


def place_undeveloped_swaths(
    m: np.ndarray,
    row_intervals,
    col_intervals,
    frontage_orientation: str,
    rng: np.random.Generator,
):
    """Merge 2-4 neighboring would-be housing blocks into each open swath.

    Only cross streets inside the merged parcel are removed. Roads along both
    long frontage sides stay intact and connect at the ends of each swath.
    """
    horizontal = frontage_orientation == "horizontal"
    runs = []
    for fixed_idx, fixed in enumerate(row_intervals if horizontal else col_intervals):
        if fixed[1] - fixed[0] != RESIDENTIAL_BLOCK_DEPTH_M:
            continue
        if horizontal:
            if fixed[0] == 0 or fixed[1] == N:
                continue
            if not (np.all(m[fixed[0] - 1, :] == ROAD)
                    and np.all(m[fixed[1], :] == ROAD)):
                continue
        else:
            if fixed[0] == 0 or fixed[1] == N:
                continue
            if not (np.all(m[:, fixed[0] - 1] == ROAD)
                    and np.all(m[:, fixed[1]] == ROAD)):
                continue

        lateral = col_intervals if horizontal else row_intervals
        for count in UNDEVELOPED_BLOCK_COUNT_OPTIONS:
            for first in range(len(lateral) - count + 1):
                part = lateral[first:first + count]
                if any(b - a not in BLOCK_LENGTH_OPTIONS_M for a, b in part):
                    continue
                # Retain a perimeter street at both lateral ends. In particular,
                # do not create an artificial full-width parcel at the map edge.
                if part[0][0] == 0 or part[-1][1] == N:
                    continue
                runs.append((fixed_idx, first, count))

    rng.shuffle(runs)
    chosen = []
    used_blocks = set()
    for fixed_idx, first, count in runs:
        occupied = {(fixed_idx, k) for k in range(first, first + count)}
        if used_blocks & occupied:
            continue

        fixed = (row_intervals if horizontal else col_intervals)[fixed_idx]
        lateral = col_intervals if horizontal else row_intervals
        start, end = lateral[first][0], lateral[first + count - 1][1]
        if horizontal:
            r0, r1, c0, c1 = fixed[0], fixed[1], start, end
        else:
            r0, r1, c0, c1 = start, end, fixed[0], fixed[1]

        # This replaces any short street segments between selected blocks.
        # It leaves perpendicular frontage streets and outer perimeter roads.
        m[r0:r1, c0:c1] = OPEN
        chosen.append({"bounds": (r0, r1, c0, c1), "blocks": count})
        used_blocks.update(occupied)
        if len(chosen) == NUM_UNDEVELOPED_SWATHS:
            break

    if len(chosen) != NUM_UNDEVELOPED_SWATHS:
        raise RuntimeError("Not enough room for the requested undeveloped swaths")
    return chosen


def place_special_blocks(
    m: np.ndarray,
    blocks,
    rng: np.random.Generator,
):
    """Reserve whole road-bounded blocks for parks and schools.

    The selected shapes follow the street grid instead of being forced to be
    squares. Every selected block must touch at least one road. We prefer blocks
    near 10,000 m^2, but broaden the size rule if needed.
    """
    candidates = []
    for block in blocks:
        r0, r1, c0, c1 = block
        if not np.all(m[r0:r1, c0:c1] == OPEN):
            continue
        road_sides = road_sides_for_region(m, r0, r1, c0, c1)
        if not road_sides:
            continue
        area_m2 = (r1 - r0) * (c1 - c0) * CELL_SIZE_M ** 2
        if SPECIAL_MIN_AREA_M2 <= area_m2 <= SPECIAL_MAX_AREA_M2:
            candidates.append(block)

    required = NUM_PARKS + NUM_SCHOOLS
    if len(candidates) < required:
        candidates = [
            block for block in blocks
            if road_sides_for_region(m, *block)
            and np.all(m[block[0]:block[1], block[2]:block[3]] == OPEN)
        ]

    if len(candidates) < required:
        raise RuntimeError("Not enough road-accessible blocks for parks/schools")

    def center(block):
        r0, r1, c0, c1 = block
        return np.array([(r0 + r1) / 2, (c0 + c1) / 2])

    def separation(a, b):
        return float(np.linalg.norm(center(a) - center(b))) * CELL_SIZE_M

    # One park per map quadrant, close to that quadrant's center. A separation
    # constraint prevents two parks close to the centerline from clustering.
    park_blocks = []
    for quadrant in rng.permutation(NUM_PARKS):
        qr, qc = divmod(int(quadrant), 2)
        target = np.array([(qr + 0.5) * N / 2, (qc + 0.5) * N / 2])
        options = [
            block for block in candidates
            if block not in park_blocks
            and int(center(block)[0] >= N / 2) == qr
            and int(center(block)[1] >= N / 2) == qc
            and all(separation(block, other) >= PARK_MIN_SEPARATION_M
                    for other in park_blocks)
        ]
        if not options:
            raise RuntimeError(f"No separated park block in quadrant {quadrant}")
        # Small seeded jitter retains variation when several blocks fit well.
        scores = [np.linalg.norm(center(b) - target) + rng.uniform(0, 15)
                  for b in options]
        park_blocks.append(options[int(np.argmin(scores))])

    # Choose the most separated pair of remaining school blocks. Distance to
    # parks breaks close scores so schools also spread away from existing parks.
    school_options = [b for b in candidates if b not in park_blocks]
    school_pair = None
    best_score = -np.inf
    for i, a in enumerate(school_options):
        for b in school_options[i + 1:]:
            d = separation(a, b)
            if d < SCHOOL_MIN_SEPARATION_M:
                continue
            park_clearance = min(separation(a, p) for p in park_blocks)
            park_clearance += min(separation(b, p) for p in park_blocks)
            score = d + 0.2 * park_clearance + rng.uniform(0, 2)
            if score > best_score:
                best_score = score
                school_pair = (a, b)
    if school_pair is None:
        raise RuntimeError("No sufficiently separated school blocks")

    chosen = park_blocks + list(school_pair)
    types = [PARK] * NUM_PARKS + [SCHOOL] * NUM_SCHOOLS

    special_areas = []
    for block, cell_type in zip(chosen, types):
        r0, r1, c0, c1 = block
        if not np.all(m[r0:r1, c0:c1] == OPEN):
            raise AssertionError("Special block was not open before placement")
        m[r0:r1, c0:c1] = cell_type
        special_areas.append({
            "bounds": block,
            "cell_type": cell_type,
            "road_sides": road_sides_for_region(m, r0, r1, c0, c1),
        })

    return special_areas


def residential_candidates_for_block(block, frontage_orientation: str):
    """Create two back-to-back rows of 20x30 m lots for one street block.

    For ``horizontal`` frontage, the block is 60 m north-south and lots face
    the top/bottom streets. For ``vertical`` frontage, the geometry is rotated.
    Internal block dimensions are generated in multiples of 20 m, so normal
    blocks fill cleanly with no large unused center.
    """
    r0, r1, c0, c1 = block
    candidates = []

    if frontage_orientation == "horizontal":
        if (r1 - r0) != RESIDENTIAL_BLOCK_DEPTH_M:
            return candidates
        frontage_length = c1 - c0
        usable = frontage_length - (frontage_length % 20)
        for c in range(c0, c0 + usable, 20):
            candidates.append((r0, r0 + 30, c, c + 20, "top", "bottom"))
            candidates.append((r1 - 30, r1, c, c + 20, "bottom", "top"))

    elif frontage_orientation == "vertical":
        if (c1 - c0) != RESIDENTIAL_BLOCK_DEPTH_M:
            return candidates
        frontage_length = r1 - r0
        usable = frontage_length - (frontage_length % 20)
        for r in range(r0, r0 + usable, 20):
            candidates.append((r, r + 20, c0, c0 + 30, "left", "right"))
            candidates.append((r, r + 20, c1 - 30, c1, "right", "left"))
    else:
        raise ValueError(f"Unknown frontage orientation: {frontage_orientation}")

    return candidates


def house_bounds_for_lot(r0, r1, c0, c1, frontage_side):
    """Place a 10x15 m house with a modest front setback and larger back yard."""
    lot_h = r1 - r0
    lot_w = c1 - c0

    if frontage_side in ("top", "bottom"):
        # 30 m deep x 20 m wide lot; house is 10 m deep x 15 m wide.
        house_h, house_w = 10, 15
        hc0 = c0 + (lot_w - house_w) // 2
        hc1 = hc0 + house_w
        if frontage_side == "top":
            hr0 = r0 + FRONT_SETBACK_M
            hr1 = hr0 + house_h
        else:
            hr1 = r1 - FRONT_SETBACK_M
            hr0 = hr1 - house_h
    else:
        # 20 m tall x 30 m deep lot; rotate the 10x15 m house.
        house_h, house_w = 15, 10
        hr0 = r0 + (lot_h - house_h) // 2
        hr1 = hr0 + house_h
        if frontage_side == "left":
            hc0 = c0 + FRONT_SETBACK_M
            hc1 = hc0 + house_w
        elif frontage_side == "right":
            hc1 = c1 - FRONT_SETBACK_M
            hc0 = hc1 - house_w
        else:
            raise ValueError(frontage_side)

    return hr0, hr1, hc0, hc1


def fill_residential_blocks(
    m: np.ndarray,
    blocks,
    frontage_orientation: str,
):
    """Fill residential blocks with back-to-back road-accessible lots."""
    residential_plots = []

    for block in blocks:
        br0, br1, bc0, bc1 = block

        # Residential blocks must be bounded by the two opposing frontage roads.
        # This excludes partial fragments at the outer map boundary, which would
        # otherwise create a row of lots without a street on one side.
        block_road_sides = set(road_sides_for_region(m, br0, br1, bc0, bc1))
        required_sides = (
            {"top", "bottom"}
            if frontage_orientation == "horizontal"
            else {"left", "right"}
        )
        if not required_sides.issubset(block_road_sides):
            continue

        candidates = residential_candidates_for_block(block, frontage_orientation)
        if not candidates:
            continue

        # A park or school consumes the whole block. Residential blocks must be
        # fully open before lots are laid out.
        if not np.all(m[br0:br1, bc0:bc1] == OPEN):
            continue

        for r0, r1, c0, c1, frontage_side, rear_side in candidates:
            road_sides = road_sides_for_region(m, r0, r1, c0, c1)
            if frontage_side not in road_sides:
                raise AssertionError(
                    f"Lot frontage {frontage_side} does not touch road: {road_sides}"
                )
            if not valid_plot_road_access(road_sides):
                raise AssertionError(f"Invalid lot road access: {road_sides}")

            hr0, hr1, hc0, hc1 = house_bounds_for_lot(
                r0, r1, c0, c1, frontage_side
            )
            m[hr0:hr1, hc0:hc1] = HOUSE

            residential_plots.append({
                "bounds": (r0, r1, c0, c1),
                "house_bounds": (hr0, hr1, hc0, hc1),
                "road_sides": road_sides,
                "frontage_side": frontage_side,
                "rear_side": rear_side,
                "block": block,
            })

    return residential_plots


def validate_residential_plots(m: np.ndarray, residential_plots):
    """Validate lot size, frontage, and back-to-back rear-yard continuity."""
    lot_mask = np.zeros(m.shape, dtype=bool)
    for plot in residential_plots:
        r0, r1, c0, c1 = plot["bounds"]
        lot_mask[r0:r1, c0:c1] = True

    for i, plot in enumerate(residential_plots):
        r0, r1, c0, c1 = plot["bounds"]
        hr0, hr1, hc0, hc1 = plot["house_bounds"]

        if sorted((r1 - r0, c1 - c0)) != [20, 30]:
            raise AssertionError(f"Plot {i} has invalid size")
        if sorted((hr1 - hr0, hc1 - hc0)) != [10, 15]:
            raise AssertionError(f"Plot {i} has invalid house size")

        road_sides = road_sides_for_region(m, r0, r1, c0, c1)
        if not valid_plot_road_access(road_sides):
            raise AssertionError(
                f"Plot {i} violates road-access rule: road sides = {road_sides}"
            )
        if plot["frontage_side"] not in road_sides:
            raise AssertionError(f"Plot {i} lost its primary street frontage")

        lot = m[r0:r1, c0:c1]
        if np.count_nonzero(lot == HOUSE) != 150:
            raise AssertionError(f"Plot {i} does not contain one 10x15 m house")
        if np.any((lot != HOUSE) & (lot != OPEN)):
            raise AssertionError(f"Plot {i} contains non-residential terrain")

        # The full rear lot line must directly abut another residential lot.
        rear = plot["rear_side"]
        if rear == "bottom":
            adjoining = lot_mask[r1, c0:c1] if r1 < N else np.array([], dtype=bool)
        elif rear == "top":
            adjoining = lot_mask[r0 - 1, c0:c1] if r0 > 0 else np.array([], dtype=bool)
        elif rear == "right":
            adjoining = lot_mask[r0:r1, c1] if c1 < N else np.array([], dtype=bool)
        elif rear == "left":
            adjoining = lot_mask[r0:r1, c0 - 1] if c0 > 0 else np.array([], dtype=bool)
        else:
            raise AssertionError(rear)

        if adjoining.size == 0 or not np.all(adjoining):
            raise AssertionError(
                f"Plot {i} rear yard does not meet another residential lot"
            )

    return True


def validate_special_areas(m: np.ndarray, special_areas, swaths):
    """Check parcel counts, road contact, and uninterrupted open swaths."""
    if sum(a["cell_type"] == PARK for a in special_areas) != NUM_PARKS:
        raise AssertionError("Incorrect number of park parcels")
    if sum(a["cell_type"] == SCHOOL for a in special_areas) != NUM_SCHOOLS:
        raise AssertionError("Incorrect number of school parcels")
    park_centers = []
    school_centers = []
    for i, area in enumerate(special_areas):
        r0, r1, c0, c1 = area["bounds"]
        road_sides = road_sides_for_region(m, r0, r1, c0, c1)
        if len(road_sides) < 1:
            raise AssertionError(f"Special area {i} does not touch a road")
        if not np.all(m[r0:r1, c0:c1] == area["cell_type"]):
            raise AssertionError(f"Special area {i} was overwritten")
        center = np.array([(r0 + r1) / 2, (c0 + c1) / 2])
        if area["cell_type"] == PARK:
            park_centers.append(center)
        else:
            school_centers.append(center)
    quadrants = {(int(p[0] >= N / 2), int(p[1] >= N / 2))
                 for p in park_centers}
    if len(quadrants) != NUM_PARKS:
        raise AssertionError("Parks are not in four different quadrants")
    for i, a in enumerate(park_centers):
        for b in park_centers[i + 1:]:
            if np.linalg.norm(a - b) * CELL_SIZE_M < PARK_MIN_SEPARATION_M:
                raise AssertionError("Park parcels are too close")
    if np.linalg.norm(school_centers[0] - school_centers[1]) * CELL_SIZE_M < SCHOOL_MIN_SEPARATION_M:
        raise AssertionError("School parcels are too close")
    if len(swaths) != NUM_UNDEVELOPED_SWATHS:
        raise AssertionError("Incorrect number of undeveloped parcels")
    for i, area in enumerate(swaths):
        r0, r1, c0, c1 = area["bounds"]
        if area["blocks"] not in UNDEVELOPED_BLOCK_COUNT_OPTIONS:
            raise AssertionError(f"Swath {i} spans an invalid number of blocks")
        if not np.all(m[r0:r1, c0:c1] == OPEN):
            raise AssertionError(f"Swath {i} is crossed by a street or building")
        if len(road_sides_for_region(m, r0, r1, c0, c1)) < 2:
            raise AssertionError(f"Swath {i} lacks surrounding road access")
    return True


def build_map(
    rng: np.random.Generator,
    return_plots: bool = False,
    return_metadata: bool = False,
):
    """Generate one constrained-random suburban 1 km x 1 km neighborhood."""
    m = np.full((N, N), OPEN, dtype=np.uint8)

    # Randomly rotate the neighborhood plan. In either orientation, residential
    # blocks are 60 m deep and 100-180 m long. This creates two back-to-back
    # rows of lots rather than a ring of houses around a large empty center.
    frontage_orientation = str(rng.choice(("horizontal", "vertical")))

    if frontage_orientation == "horizontal":
        horizontal_roads = make_road_bands(rng, (RESIDENTIAL_BLOCK_DEPTH_M,))
        vertical_roads = make_road_bands(rng, BLOCK_LENGTH_OPTIONS_M)
    else:
        horizontal_roads = make_road_bands(rng, BLOCK_LENGTH_OPTIONS_M)
        vertical_roads = make_road_bands(rng, (RESIDENTIAL_BLOCK_DEPTH_M,))

    for start, end in horizontal_roads:
        m[start:end, :] = ROAD
    for start, end in vertical_roads:
        m[:, start:end] = ROAD

    row_intervals = free_intervals(horizontal_roads)
    col_intervals = free_intervals(vertical_roads)
    blocks = [
        (r0, r1, c0, c1)
        for r0, r1 in row_intervals
        for c0, c1 in col_intervals
    ]

    swaths = place_undeveloped_swaths(
        m, row_intervals, col_intervals, frontage_orientation, rng
    )

    # Swaths use OPEN cells, just like yards. Reserve the original housing
    # blocks they cover so they cannot be filled with houses or special parcels.
    def intersects(block, swath):
        r0, r1, c0, c1 = block
        sr0, sr1, sc0, sc1 = swath["bounds"]
        return r0 < sr1 and sr0 < r1 and c0 < sc1 and sc0 < c1

    buildable_blocks = [
        block for block in blocks
        if not any(intersects(block, swath) for swath in swaths)
    ]

    # Parks and schools consume whole road-bounded blocks. This naturally makes
    # them rectangular/non-square when street spacing is rectangular, and every
    # selected parcel is guaranteed to touch at least one road.
    special_areas = place_special_blocks(m, buildable_blocks, rng)

    residential_plots = fill_residential_blocks(
        m, buildable_blocks, frontage_orientation
    )
    validate_residential_plots(m, residential_plots)
    validate_special_areas(m, special_areas, swaths)

    if return_metadata:
        return m, residential_plots, special_areas, swaths
    if return_plots:
        return m, residential_plots
    return m


# -----------------------------------------------------------------------------
# People / dynamic hazard generation
# -----------------------------------------------------------------------------

def backyard_regions_and_mask(base_map: np.ndarray, residential_plots):
    """Identify the open area behind each house, one region per residential lot."""
    regions = []
    mask = np.zeros(base_map.shape, dtype=bool)
    for plot in residential_plots:
        r0, r1, c0, c1 = plot["bounds"]
        hr0, hr1, hc0, hc1 = plot["house_bounds"]
        side = plot["frontage_side"]
        if side == "top":
            region = (hr1, r1, c0, c1)
        elif side == "bottom":
            region = (r0, hr0, c0, c1)
        elif side == "left":
            region = (r0, r1, hc1, c1)
        else:
            region = (r0, r1, c0, hc0)
        a, b, x, y = region
        if a >= b or x >= y or not np.all(base_map[a:b, x:y] == OPEN):
            raise AssertionError("Backyard is not a nonempty open region")
        regions.append(region)
        mask[a:b, x:y] = True
    return regions, mask


def find_person_center_candidates(
    base_map: np.ndarray, backyard_mask: np.ndarray | None = None
):
    """Find park, sidewalk, and school-edge centers (excluding backyards)."""
    open_mask = base_map == OPEN
    road_mask = base_map == ROAD
    park_mask = base_map == PARK
    school_mask = base_map == SCHOOL

    def four_neighbors(mask):
        adjacent = np.zeros_like(mask)
        adjacent[1:, :] |= mask[:-1, :]
        adjacent[:-1, :] |= mask[1:, :]
        adjacent[:, 1:] |= mask[:, :-1]
        adjacent[:, :-1] |= mask[:, 1:]
        return adjacent

    eligible = (
        park_mask
        | (open_mask & four_neighbors(road_mask))
        | (road_mask & four_neighbors(open_mask | school_mask))
        | (school_mask & four_neighbors(road_mask))
    )
    if backyard_mask is not None:
        if backyard_mask.shape != base_map.shape:
            raise ValueError("Backyard mask must match the map")
        eligible &= ~backyard_mask

    # A 5x5 m danger zone needs two cells of room around its center.
    radius = PERSON_DANGER_SIZE_M // 2
    eligible[:radius, :] = False
    eligible[-radius:, :] = False
    eligible[:, :radius] = False
    eligible[:, -radius:] = False

    return np.argwhere(eligible)


def place_people(
    base_map: np.ndarray,
    candidate_centers: np.ndarray,
    rng: np.random.Generator,
    count: int = PEOPLE_PER_TRIAL,
    backyard_regions=(),
    park_regions=(),
):
    """Sample street edges, capped parks, and sparsely occupied backyards."""
    if not park_regions and np.any(base_map == PARK):
        raise ValueError("Park bounds are required to enforce the per-park cap")
    backyard_share = BACKYARD_PERSON_FRACTION if backyard_regions else 0.0
    park_share = PARK_PERSON_FRACTION if park_regions else 0.0
    backyard_count, park_target, _ = rng.multinomial(
        count, [backyard_share, park_share, 1 - backyard_share - park_share]
    )
    backyard_count = int(backyard_count)
    park_target = min(int(park_target), int(count * PARK_PERSON_FRACTION))
    if backyard_count > MAX_PEOPLE_PER_BACKYARD * len(backyard_regions):
        raise RuntimeError("Not enough backyard capacity for selected people")

    park_cells = base_map[candidate_centers[:, 0], candidate_centers[:, 1]] == PARK
    public_candidates = candidate_centers[~park_cells]
    park_candidates = []
    for r0, r1, c0, c1 in park_regions:
        subset = candidate_centers[
            park_cells
            & (candidate_centers[:, 0] >= r0)
            & (candidate_centers[:, 0] < r1)
            & (candidate_centers[:, 1] >= c0)
            & (candidate_centers[:, 1] < c1)
        ]
        park_candidates.append(subset)
    per_park_cap = (
        math.ceil(count * PARK_PERSON_FRACTION / len(park_regions))
        if park_regions else 0
    )
    park_caps = np.array(
        [min(per_park_cap, len(x)) for x in park_candidates], dtype=int
    )
    park_counts = np.zeros(len(park_regions), dtype=int)
    for _ in range(min(park_target, int(park_caps.sum()))):
        available = np.flatnonzero(park_counts < park_caps)
        park_counts[int(rng.choice(available))] += 1

    public_count = count - backyard_count - int(park_counts.sum())
    if len(public_candidates) < public_count:
        raise RuntimeError(
            f"Only {len(public_candidates)} non-park public centers exist, "
            f"but {public_count} public centers were requested."
        )

    selected = rng.choice(len(public_candidates), size=public_count, replace=False)
    center_groups = [public_candidates[selected]]
    for subset, park_count in zip(park_candidates, park_counts):
        selected = rng.choice(len(subset), size=int(park_count), replace=False)
        center_groups.append(subset[selected])
    radius = PERSON_DANGER_SIZE_M // 2
    backyard_centers = []
    # Each lot contributes two selectable slots. Sampling slots without
    # replacement allows zero, one, or two people in each backyard.
    slots = rng.choice(
        MAX_PEOPLE_PER_BACKYARD * len(backyard_regions),
        size=backyard_count, replace=False,
    )
    lot_indices, occupancy = np.unique(
        slots // MAX_PEOPLE_PER_BACKYARD, return_counts=True
    )
    for index, residents in zip(lot_indices, occupancy):
        r0, r1, c0, c1 = backyard_regions[index]
        # Keep the 5x5 m danger zone entirely inside the map; the center is
        # inside its selected backyard. Two residents get distinct centers.
        row_start, row_end = max(r0, radius), min(r1, N - radius)
        col_start, col_end = max(c0, radius), min(c1, N - radius)
        width = col_end - col_start
        choices = rng.choice((row_end - row_start) * width,
                             size=int(residents), replace=False)
        for offset in choices:
            backyard_centers.append((row_start + int(offset // width),
                                     col_start + int(offset % width)))
    if backyard_centers:
        center_groups.append(np.asarray(backyard_centers, dtype=int))
    centers = np.vstack(center_groups)

    hazard_mask = np.zeros(base_map.shape, dtype=bool)

    for r, c in centers:
        hazard_mask[
            r - radius:r + radius + 1,
            c - radius:c + radius + 1,
        ] = True

    return hazard_mask, centers, backyard_count, park_counts


# -----------------------------------------------------------------------------
# Landing planner
# -----------------------------------------------------------------------------

def actual_cell_type(base_map: np.ndarray, person_mask: np.ndarray, pos):
    """Return PERSON if the point is in a person danger zone; otherwise terrain."""
    if person_mask[pos]:
        return PERSON
    return int(base_map[pos])


def ranked_reachable_sites(base_map: np.ndarray, pos, range_m: float):
    """Return up to 5 distinct low-score reachable sites in best-first order."""
    r, c = pos
    radius_cells = range_m / CELL_SIZE_M
    radius_int = int(math.ceil(radius_cells))

    # Only examine a local bounding box instead of all 1,000,000 cells.
    r0 = max(0, r - radius_int)
    r1 = min(N, r + radius_int + 1)
    c0 = max(0, c - radius_int)
    c1 = min(N, c + radius_int + 1)

    rr, cc = np.ogrid[r0:r1, c0:c1]
    distance_cells = np.hypot(rr - r, cc - c)
    reachable = distance_cells <= radius_cells

    local_types = base_map[r0:r1, c0:c1]
    distance_m = distance_cells * CELL_SIZE_M

    # Planner knows only the static map, not the current people locations.
    score = RISK[local_types] + DISTANCE_WEIGHT_PER_M * distance_m

    flat_score = score.ravel()
    valid = np.flatnonzero(reachable.ravel())

    if len(valid) == 0:
        return [pos]

    # Find a modest pool of best cells without sorting the whole reachable area.
    pool_size = min(CANDIDATE_POOL, len(valid))
    valid_scores = flat_score[valid]

    if pool_size < len(valid):
        selected = np.argpartition(valid_scores, pool_size - 1)[:pool_size]
        pool = valid[selected]
    else:
        pool = valid

    pool = pool[np.argsort(flat_score[pool])]

    local_width = c1 - c0
    min_sep_cells = RESELECT_MIN_SEPARATION_M / CELL_SIZE_M
    min_sep_sq = min_sep_cells ** 2

    candidates = []
    for flat_idx in pool:
        local_r = int(flat_idx // local_width)
        local_c = int(flat_idx % local_width)
        cand = (r0 + local_r, c0 + local_c)

        # With 1 m cells, neighboring cells should not count as five genuinely
        # different landing sites. Enforce a small separation between choices.
        if all(
            (cand[0] - prev[0]) ** 2 + (cand[1] - prev[1]) ** 2 >= min_sep_sq
            for prev in candidates
        ):
            candidates.append(cand)
            if len(candidates) >= VERIFY_CANDIDATES:
                break

    if not candidates:
        candidates.append(pos)

    return candidates


# -----------------------------------------------------------------------------
# Plotting and reporting
# -----------------------------------------------------------------------------

def save_map_image(
    base_map: np.ndarray,
    map_name: str,
    output_path: Path,
    show: bool = SHOW_MAP_IMAGES,
    residential_plots=None,
    person_mask: np.ndarray | None = None,
    person_centers: np.ndarray | None = None,
):
    """Save a static map, optionally overlaying one trial's people and hazards."""
    if (person_mask is None) != (person_centers is None):
        raise ValueError("person_mask and person_centers must be supplied together")
    colors = [
        "#cfe8b4",  # open
        "#4f9d4a",  # park
        "#666666",  # road
        "#d8a66f",  # house
        "#4f86d9",  # school
        "#d62728",  # person (not present in static image)
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        base_map,
        cmap=cmap,
        vmin=OPEN,
        vmax=PERSON,
        interpolation="nearest",
        origin="upper",
    )

    if person_mask is not None:
        if person_mask.shape != base_map.shape:
            raise ValueError("Person hazard mask must match the map")
        overlay = np.ma.masked_where(~person_mask, person_mask)
        ax.imshow(
            overlay, cmap=ListedColormap(["#dd202c"]), vmin=0, vmax=1,
            alpha=0.60, interpolation="nearest", origin="upper",
        )
        ax.scatter(
            person_centers[:, 1], person_centers[:, 0],
            s=3, c="#270c16", alpha=0.85, linewidths=0, zorder=4,
        )
        ax.set_title(
            f"{map_name} - {len(person_centers):,} Person Centers and 5 x 5 m Danger Zones"
        )
    else:
        ax.set_title(f"{map_name} - Static Neighborhood Design")
    ax.set_xlabel("Meters east-west")
    ax.set_ylabel("Meters north-south")

    tick_step = 100
    ticks = np.arange(0, N + 1, tick_step)
    ticks = ticks[ticks < N]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels((ticks * CELL_SIZE_M).astype(int))
    ax.set_yticklabels((ticks * CELL_SIZE_M).astype(int))

    legend = [
        Patch(facecolor=colors[OPEN], label="Open / yard / undeveloped"),
        Patch(facecolor=colors[PARK], label="Park"),
        Patch(facecolor=colors[ROAD], label="Road"),
        Patch(facecolor=colors[HOUSE], label="House"),
        Patch(facecolor=colors[SCHOOL], label="School"),
    ]
    if person_mask is not None:
        legend.extend([
            Patch(facecolor="#dd202c", alpha=0.60, label="Person danger zone (5 x 5 m)"),
            Line2D([0], [0], marker=".", linestyle="None", color="#270c16",
                   markersize=7, label="Person center"),
        ])
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # Draw thin lot boundaries so the two back-to-back rows are visible.
    if residential_plots:
        for plot in residential_plots:
            r0, r1, c0, c1 = plot["bounds"]
            ax.add_patch(Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.35,
            ))

    # 100 m scale bar near the bottom-left.
    bar_y = N - 35
    bar_x0 = 35
    bar_x1 = bar_x0 + int(100 / CELL_SIZE_M)
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="black", linewidth=4)
    ax.text((bar_x0 + bar_x1) / 2, bar_y - 12, "100 m", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def print_composition_table(static_pct, effective_pct, context):
    """Explain area shares and the dynamic hazard overlay in one table."""
    print(f"LAND COVERAGE (% of map cells; {context})")
    print("  Static = terrain on the map; effective = terrain after current 5×5 m")
    print("  person danger zones override covered cells, averaged over trials.")
    print(f"  {'Cell type':<13} {'Static %':>10} {'Effective %':>13}")
    for cell_type in CELL_TYPES:
        print(
            f"  {CELL_NAMES[cell_type]:<13} "
            f"{static_pct[cell_type]:10.2f} {effective_pct[cell_type]:13.2f}"
        )


def print_landing_table(counts, high_counts, risk_sums, trials):
    """Each strategy has exactly one landing per simulated failure."""
    print(f"LANDINGS (% of {trials} landings PER strategy)")
    print("  Each cell-type column is the ACTUAL landing surface. 'Person' means")
    print("  a 5×5 m danger zone covered the site and overrides its terrain type.")
    print("  High risk = actual risk ≥ 0.60 (road, house, school, or person).")
    print("  Mean risk = average assigned score, not an accident probability.")
    header = f"  {'Strategy':<20}" + "".join(
        f"{CELL_NAMES[t]:>9}" for t in CELL_TYPES
    ) + f"{'High risk':>12}{'Mean risk':>12}"
    print(header)
    for strategy in STRATEGIES:
        values = "".join(
            f"{pct(counts[strategy][t], trials):8.1f}%" for t in CELL_TYPES
        )
        print(
            f"  {STRATEGY_LABELS[strategy]:<20}{values}"
            f"{pct(high_counts[strategy], trials):11.1f}%"
            f"{risk_sums[strategy] / trials:12.4f}"
        )


def save_statistics_chart(
    title: str,
    output_path: Path,
    static_pct: np.ndarray,
    effective_pct: np.ndarray,
    landing_counts,
    high_counts,
    risk_sums,
    trials: int,
):
    """Save the four reported statistics as labeled bar panels."""
    colors = ["#cfe8b4", "#4f9d4a", "#666666", "#d8a66f",
              "#4f86d9", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_area, ax_land, ax_high, ax_risk = axes.ravel()
    fig.suptitle(f"{title} — landing simulation results", fontsize=17, y=0.975)

    categories = ["Static map", "With people (avg.)"]
    left = np.zeros(2)
    for t in CELL_TYPES:
        widths = [static_pct[t], effective_pct[t]]
        ax_area.barh(categories, widths, left=left, color=colors[t],
                     label=CELL_NAMES[t].capitalize())
        left += widths
    ax_area.invert_yaxis()
    ax_area.set(xlim=(0, 100), xlabel="Share of map cells (%)",
                title="Land coverage — denominator: 1,000,000 cells per map")

    strategy_names = [STRATEGY_LABELS[s] for s in STRATEGIES]
    left = np.zeros(len(STRATEGIES))
    for t in CELL_TYPES:
        widths = [pct(landing_counts[s][t], trials) for s in STRATEGIES]
        ax_land.barh(strategy_names, widths, left=left, color=colors[t])
        left += widths
    ax_land.invert_yaxis()
    ax_land.set(xlim=(0, 100), xlabel="Share of landings (%)",
                title=f"Actual landing surface — {trials} landings per strategy")

    high_values = [pct(high_counts[s], trials) for s in STRATEGIES]
    bars = ax_high.bar(strategy_names, high_values,
                       color=["#526a89", "#718db0", "#2d746e"])
    ax_high.bar_label(bars, fmt="%.1f%%", padding=3)
    ax_high.set_ylim(0, min(105, max(8, max(high_values) * 1.22)))
    ax_high.set(ylabel="High-risk landings (%)",
                title="Road, house, school, or person hazard")

    risk_values = [risk_sums[s] / trials for s in STRATEGIES]
    bars = ax_risk.bar(strategy_names, risk_values,
                       color=["#526a89", "#718db0", "#2d746e"])
    ax_risk.bar_label(bars, fmt="%.3f", padding=3)
    ax_risk.set_ylim(0, max(0.10, max(risk_values) * 1.22))
    ax_risk.set(ylabel="Mean assigned risk score",
                title="Average score at the actual landing cell")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax in (ax_area, ax_land) else "y",
                alpha=0.18)
        ax.set_axisbelow(True)
    for ax in (ax_high, ax_risk):
        ax.tick_params(axis="x", rotation=12)

    handles = [Patch(facecolor=colors[t], label=CELL_NAMES[t].capitalize())
               for t in CELL_TYPES]
    fig.legend(handles=handles, loc="lower center", ncol=len(CELL_TYPES),
               bbox_to_anchor=(0.5, 0.045), frameon=False)
    fig.text(0.5, 0.012,
             "Person danger zones override terrain. Risk scores are model weights, not probabilities.",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.94), h_pad=3.0, w_pad=3.0)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Monte Carlo experiment
# -----------------------------------------------------------------------------

def _run_experiment(
    num_maps: int = NUM_MAPS,
    trials_per_map: int = TRIALS_PER_MAP,
):
    if num_maps < 1 or trials_per_map < 1:
        raise ValueError("At least one map and one trial per map are required")
    OUTPUT_DIR.mkdir(exist_ok=True)
    MAP_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    print("SAFE2DITCH-STYLE EMERGENCY LANDING SIMULATION")
    print(f"{num_maps} independently generated 1 km² maps; "
          f"{trials_per_map} drone failures on each map.")
    print("The same failure and person positions are used to compare all three")
    print("strategies within a trial. Each strategy makes one landing per trial.")
    print("Drop in place = current cell; Map only = best reachable mapped site;")
    print("Map + verification = inspect up to five distinct candidate sites.")
    print(f"Each trial places {PEOPLE_PER_TRIAL} people; roughly "
          f"{BACKYARD_PERSON_FRACTION:.0%} are sampled from backyards")
    print(f"(at most {MAX_PEOPLE_PER_BACKYARD} per yard). Parks are capped "
          f"at {PARK_PERSON_FRACTION:.0%} "
          f"of all people, or {int(PEOPLE_PER_TRIAL * PARK_PERSON_FRACTION)} total")
    print(f"across {NUM_PARKS} parks (at most "
          f"{math.ceil(PEOPLE_PER_TRIAL * PARK_PERSON_FRACTION / NUM_PARKS)} "
          "in any single park).")
    print("The others use road/open and road/school borders.")
    print("Each saved map image displays its FIRST trial's people; the people")
    print("are redrawn in the remaining trials used for the statistics.")
    print("Percentages describe shares of cells or simulated landings, not")
    print("probabilities of injury. OPEN includes yards and undeveloped land.")

    # Spawn separate deterministic RNG streams for each map and each map's trials.
    map_seed_sequence = np.random.SeedSequence(MAP_MASTER_SEED)
    trial_seed_sequence = np.random.SeedSequence(TRIAL_MASTER_SEED)
    map_seeds = map_seed_sequence.spawn(num_maps)
    trial_seeds = trial_seed_sequence.spawn(num_maps)

    overall_landing_counts = {
        strategy: np.zeros(len(CELL_TYPES), dtype=np.int64)
        for strategy in STRATEGIES
    }
    overall_high_counts = {strategy: 0 for strategy in STRATEGIES}
    overall_risk_sums = {strategy: 0.0 for strategy in STRATEGIES}

    overall_static_counts = np.zeros(len(CELL_TYPES), dtype=np.int64)
    overall_effective_counts = np.zeros(len(CELL_TYPES), dtype=np.int64)

    composition_rows = []
    landing_rows = []
    strategy_rows_by_map = []
    design_rows = []

    for map_index in range(num_maps):
        map_number = map_index + 1
        map_name = f"Map_{map_number:02d}"

        map_rng = np.random.default_rng(map_seeds[map_index])
        trial_rng = np.random.default_rng(trial_seeds[map_index])

        base_map, residential_plots, special_areas, swaths = build_map(
            map_rng, return_metadata=True
        )
        park_regions = [a["bounds"] for a in special_areas if a["cell_type"] == PARK]
        backyard_regions, backyard_mask = backyard_regions_and_mask(
            base_map, residential_plots
        )
        person_candidates = find_person_center_candidates(base_map, backyard_mask)

        image_path = MAP_DIR / f"{map_name}.png"

        print("\n" + "=" * 72)
        one_side_plots = sum(len(p["road_sides"]) == 1 for p in residential_plots)
        two_side_plots = sum(len(p["road_sides"]) == 2 for p in residential_plots)

        print(f"Generated {map_name}")
        print(f"Map image with trial 1 people: {image_path}")
        print(
            f"Parcels: {NUM_PARKS} parks, {NUM_SCHOOLS} schools, "
            f"{len(swaths)} undeveloped swaths spanning "
            + ", ".join(str(a["blocks"]) for a in swaths)
            + " blocks"
        )
        def parcel_center(area):
            r0, r1, c0, c1 = area["bounds"]
            return np.array([(r0 + r1) / 2, (c0 + c1) / 2])

        park_centers = [parcel_center(a) for a in special_areas
                        if a["cell_type"] == PARK]
        school_centers = [parcel_center(a) for a in special_areas
                          if a["cell_type"] == SCHOOL]
        park_min_distance = min(
            np.linalg.norm(a - b) * CELL_SIZE_M
            for i, a in enumerate(park_centers) for b in park_centers[i + 1:]
        )
        school_distance = float(
            np.linalg.norm(school_centers[0] - school_centers[1]) * CELL_SIZE_M
        )
        print(
            f"Nearest two parks: {park_min_distance:.0f} m; "
            f"schools: {school_distance:.0f} m apart"
        )
        design_rows.append({
            "map": map_name,
            "parks": sum(a["cell_type"] == PARK for a in special_areas),
            "schools": sum(a["cell_type"] == SCHOOL for a in special_areas),
            "undeveloped_swaths": len(swaths),
            "undeveloped_blocks_total": sum(a["blocks"] for a in swaths),
            "undeveloped_area_m2": sum(
                (a["bounds"][1] - a["bounds"][0])
                * (a["bounds"][3] - a["bounds"][2]) for a in swaths
            ),
            "residential_plots": len(residential_plots),
            "min_park_separation_m": round(float(park_min_distance), 1),
            "school_separation_m": round(school_distance, 1),
        })
        print(
            f"Residential plots: {len(residential_plots)} "
            f"({one_side_plots} with 1 road side, "
            f"{two_side_plots} corner plots with 2 adjacent road sides)"
        )
        print(f"Running {trials_per_map} failure trials on {map_name}...")

        static_counts = np.bincount(
            base_map.ravel(), minlength=len(CELL_TYPES)
        ).astype(np.int64)
        overall_static_counts += static_counts

        static_pct = np.zeros(len(CELL_TYPES), dtype=float)
        static_pct[:len(STATIC_CELL_TYPES)] = (
            100.0 * static_counts[:len(STATIC_CELL_TYPES)] / base_map.size
        )

        map_landing_counts = {
            strategy: np.zeros(len(CELL_TYPES), dtype=np.int64)
            for strategy in STRATEGIES
        }
        map_high_counts = {strategy: 0 for strategy in STRATEGIES}
        map_risk_sums = {strategy: 0.0 for strategy in STRATEGIES}

        # Effective composition includes person danger zones. Because people move
        # every trial, accumulate composition and average it after all 100 trials.
        effective_counts_sum = np.zeros(len(CELL_TYPES), dtype=np.int64)
        backyard_count_sum = 0
        park_count_sum = 0
        maximum_park_count = 0

        for trial in range(trials_per_map):
            person_mask, person_centers, backyard_count, park_counts = place_people(
                base_map,
                person_candidates,
                trial_rng,
                count=PEOPLE_PER_TRIAL,
                backyard_regions=backyard_regions,
                park_regions=park_regions,
            )
            backyard_count_sum += backyard_count
            park_count_sum += int(park_counts.sum())
            maximum_park_count = max(maximum_park_count, int(park_counts.max()))
            if trial == 0:
                save_map_image(
                    base_map, map_name, image_path,
                    residential_plots=residential_plots,
                    person_mask=person_mask, person_centers=person_centers,
                )

            # Compute dynamic/effective map composition efficiently.
            # Person danger zones override the underlying terrain for landing risk.
            covered_base_counts = np.bincount(
                base_map[person_mask], minlength=len(CELL_TYPES)
            ).astype(np.int64)

            effective_counts = static_counts.copy()
            effective_counts[:len(STATIC_CELL_TYPES)] -= (
                covered_base_counts[:len(STATIC_CELL_TYPES)]
            )
            effective_counts[PERSON] = int(person_mask.sum())
            effective_counts_sum += effective_counts

            # Random failure point and remaining reachable range.
            pos = tuple(int(x) for x in trial_rng.integers(0, N, size=2))
            remaining_range_m = float(
                trial_rng.uniform(MIN_RANGE_M, MAX_RANGE_M)
            )

            # BASELINE: land directly below current position.
            baseline_choice = pos

            # Safe2Ditch planner ranks sites using only the static map.
            candidates = ranked_reachable_sites(
                base_map,
                pos,
                remaining_range_m,
            )

            # No-verification variant takes the planner's first choice.
            noverify_choice = candidates[0]

            # Verification variant checks actual conditions and can re-select
            # up to four times (five total candidates).
            verified_choice = candidates[-1]
            for cand in candidates[:VERIFY_CANDIDATES]:
                verified_choice = cand
                cand_type = actual_cell_type(base_map, person_mask, cand)
                if RISK[cand_type] < HIGH:
                    break

            choices = {
                "baseline": baseline_choice,
                "s2d_noverify": noverify_choice,
                "s2d": verified_choice,
            }

            for strategy, choice in choices.items():
                landing_type = actual_cell_type(base_map, person_mask, choice)
                landing_risk = float(RISK[landing_type])

                map_landing_counts[strategy][landing_type] += 1
                map_risk_sums[strategy] += landing_risk
                map_high_counts[strategy] += int(landing_risk >= HIGH)

                overall_landing_counts[strategy][landing_type] += 1
                overall_risk_sums[strategy] += landing_risk
                overall_high_counts[strategy] += int(landing_risk >= HIGH)

        overall_effective_counts += effective_counts_sum

        avg_effective_pct = (
            100.0 * effective_counts_sum
            / (trials_per_map * base_map.size)
        )
        print_composition_table(
            static_pct, avg_effective_pct,
            f"{base_map.size:,} cells of 1 m² each; effective share averaged over {trials_per_map} trials",
        )
        print_landing_table(
            map_landing_counts, map_high_counts, map_risk_sums, trials_per_map
        )
        print(
            f"Backyard people per trial: {backyard_count_sum / trials_per_map:.1f} "
            f"of {PEOPLE_PER_TRIAL} on average, across {len(backyard_regions)} lots."
        )
        print(
            f"Park people per trial: {park_count_sum / trials_per_map:.1f} "
            f"across {len(park_regions)} parks on average; maximum observed "
            f"in any one park: {maximum_park_count} "
            f"(cap {math.ceil(PEOPLE_PER_TRIAL * PARK_PERSON_FRACTION / len(park_regions))})."
        )
        chart_path = CHART_DIR / f"{map_name}_statistics.png"
        save_statistics_chart(
            map_name, chart_path, static_pct, avg_effective_pct,
            map_landing_counts, map_high_counts, map_risk_sums, trials_per_map,
        )
        print(f"Statistics chart: {chart_path}")

        for strategy in STRATEGIES:
            high_pct = pct(map_high_counts[strategy], trials_per_map)
            mean_risk = map_risk_sums[strategy] / trials_per_map
            for cell_type in CELL_TYPES:
                strike_pct = pct(
                    map_landing_counts[strategy][cell_type],
                    trials_per_map,
                )

                landing_rows.append({
                    "map": map_name,
                    "strategy": strategy,
                    "cell_type": CELL_NAMES[cell_type],
                    "landing_count": int(
                        map_landing_counts[strategy][cell_type]
                    ),
                    "landing_pct": strike_pct,
                })

            strategy_rows_by_map.append({
                "map": map_name,
                "strategy": strategy,
                "high_risk_pct": high_pct,
                "mean_risk": mean_risk,
            })

        for cell_type in CELL_TYPES:
            composition_rows.append({
                "map": map_name,
                "cell_type": CELL_NAMES[cell_type],
                "static_map_pct": static_pct[cell_type],
                "avg_effective_map_pct": avg_effective_pct[cell_type],
            })

    # -------------------------------------------------------------------------
    # Overall results across all maps and all trials
    # -------------------------------------------------------------------------
    total_trials = num_maps * trials_per_map
    total_static_cells = num_maps * N * N
    total_effective_cell_observations = total_trials * N * N

    print("\n" + "=" * 72)
    print(
        f"OVERALL RESULTS: {num_maps} maps x {trials_per_map} trials "
        f"= {total_trials} failures"
    )

    overall_static_pct = 100.0 * overall_static_counts / total_static_cells
    overall_effective_pct = (
        100.0 * overall_effective_counts / total_effective_cell_observations
    )

    print_composition_table(
        overall_static_pct, overall_effective_pct,
        f"{N*N:,} cells per map; static averaged over {num_maps} maps, effective over {total_trials} trials",
    )
    print_landing_table(
        overall_landing_counts, overall_high_counts, overall_risk_sums,
        total_trials,
    )
    overall_chart_path = CHART_DIR / "Overall_statistics.png"
    save_statistics_chart(
        "Overall", overall_chart_path, overall_static_pct,
        overall_effective_pct, overall_landing_counts, overall_high_counts,
        overall_risk_sums, total_trials,
    )
    print(f"Statistics chart: {overall_chart_path}")

    overall_strategy_rows = []

    for strategy in STRATEGIES:
        high_pct = pct(overall_high_counts[strategy], total_trials)
        mean_risk = overall_risk_sums[strategy] / total_trials

        row = {
            "strategy": strategy,
            "high_risk_pct": high_pct,
            "mean_risk": mean_risk,
        }

        for cell_type in CELL_TYPES:
            landing_pct = pct(
                overall_landing_counts[strategy][cell_type],
                total_trials,
            )
            row[f"{CELL_NAMES[cell_type]}_landing_pct"] = landing_pct

        overall_strategy_rows.append(row)

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------
    write_csv(
        OUTPUT_DIR / "map_composition.csv",
        composition_rows,
        [
            "map",
            "cell_type",
            "static_map_pct",
            "avg_effective_map_pct",
        ],
    )

    write_csv(
        OUTPUT_DIR / "map_design_summary.csv",
        design_rows,
        ["map", "parks", "schools", "undeveloped_swaths",
         "undeveloped_blocks_total", "undeveloped_area_m2",
         "residential_plots", "min_park_separation_m", "school_separation_m"],
    )

    write_csv(
        OUTPUT_DIR / "landing_outcomes_by_map.csv",
        landing_rows,
        [
            "map",
            "strategy",
            "cell_type",
            "landing_count",
            "landing_pct",
        ],
    )

    write_csv(
        OUTPUT_DIR / "strategy_summary_by_map.csv",
        strategy_rows_by_map,
        ["map", "strategy", "high_risk_pct", "mean_risk"],
    )

    overall_fields = [
        "strategy",
        "high_risk_pct",
        "mean_risk",
    ] + [f"{CELL_NAMES[t]}_landing_pct" for t in CELL_TYPES]

    write_csv(
        OUTPUT_DIR / "strategy_summary_overall.csv",
        overall_strategy_rows,
        overall_fields,
    )

    print("\nSaved outputs:")
    print(f"  Map images:                   {MAP_DIR}/Map_01.png ... Map_{num_maps:02d}.png")
    print(f"  Statistics charts:            {CHART_DIR}/Map_01_statistics.png ... Overall_statistics.png")
    print(f"  This report:                  {REPORT_PATH}")
    print(f"  Map composition:              {OUTPUT_DIR / 'map_composition.csv'}")
    print(f"  Parcel counts:                {OUTPUT_DIR / 'map_design_summary.csv'}")
    print(f"  Landing outcomes by map:      {OUTPUT_DIR / 'landing_outcomes_by_map.csv'}")
    print(f"  Strategy summary by map:      {OUTPUT_DIR / 'strategy_summary_by_map.csv'}")
    print(f"  Overall strategy summary:     {OUTPUT_DIR / 'strategy_summary_overall.csv'}")

    return {
        "overall_landing_counts": overall_landing_counts,
        "overall_high_counts": overall_high_counts,
        "overall_risk_sums": overall_risk_sums,
        "overall_static_pct": overall_static_pct,
        "overall_effective_pct": overall_effective_pct,
    }


def run_experiment(
    num_maps: int = NUM_MAPS,
    trials_per_map: int = TRIALS_PER_MAP,
):
    """Write the full human-readable report to disk without console spam."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(report):
            return _run_experiment(num_maps, trials_per_map)


def generate_people_map(map_number: int = 1, output_path: Path | None = None):
    """Render a selected seeded map with the people from its first trial.

    Separate map/trial seed streams make this preview reproducible without
    changing the default Monte Carlo results or writing the experiment log.
    """
    if not 1 <= map_number <= NUM_MAPS:
        raise ValueError(f"Map number must be between 1 and {NUM_MAPS}")
    map_seed = np.random.SeedSequence(MAP_MASTER_SEED).spawn(NUM_MAPS)[map_number - 1]
    trial_seed = np.random.SeedSequence(TRIAL_MASTER_SEED).spawn(NUM_MAPS)[map_number - 1]
    base_map, plots, special_areas, _ = build_map(
        np.random.default_rng(map_seed), return_metadata=True
    )
    park_regions = [a["bounds"] for a in special_areas if a["cell_type"] == PARK]
    backyard_regions, backyard_mask = backyard_regions_and_mask(base_map, plots)
    centers_allowed = find_person_center_candidates(base_map, backyard_mask)
    mask, centers, _, _ = place_people(
        base_map, centers_allowed, np.random.default_rng(trial_seed),
        backyard_regions=backyard_regions, park_regions=park_regions,
    )
    path = (Path(output_path) if output_path is not None
            else MAP_DIR / f"Map_{map_number:02d}_with_people.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_map_image(
        base_map, f"Map_{map_number:02d}", path,
        residential_plots=plots, person_mask=mask, person_centers=centers,
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--people-map", action="store_true",
        help="Save one map showing person centers and their 5 x 5 m danger zones."
    )
    parser.add_argument(
        "--map-number", type=int, default=1, choices=range(1, NUM_MAPS + 1),
        metavar=f"1-{NUM_MAPS}", help="Which seeded map to preview (default: 1)."
    )
    parser.add_argument(
        "--output", type=Path, help="PNG output path for --people-map."
    )
    args = parser.parse_args()
    if args.people_map:
        path = generate_people_map(args.map_number, args.output)
        print(f"Saved map with people to {path}")
    else:
        if args.output is not None or args.map_number != 1:
            parser.error("--output and --map-number require --people-map")
        run_experiment()
