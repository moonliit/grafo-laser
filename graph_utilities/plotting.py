# corrected animate_walk_simple with reliable fading
from typing import List, Tuple, Dict, Optional, Any, cast
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

import math
import networkx as nx
import matplotlib.pyplot as plt

def _compute_pos2d_from_graph(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    pos3d = nx.get_node_attributes(G, "pos")
    if pos3d and len(pos3d) == G.number_of_nodes():
        return {k: (v[0], v[1]) for k, v in pos3d.items()}
    return nx.spring_layout(G, seed=42, iterations=200)


def animate_walk_simple(
    G: nx.Graph,
    walk: List[int],
    interval: int = 250,
    arrow_every: int = 3,
    cmap_name: str = "plasma",
    node_size: int = 300,
    figsize: Tuple[int, int] = (6, 6),
    show_edge_weights: bool = True,
    repeat: bool = True,
    annotate_nodes_during_play: bool = True,
    annotate_first_visit_only: bool = False,
    repeat_offset_scale: float = 0.03,
    repeat_linewidth_base: float = 2.8,
    repeat_linewidth_step: float = 0.8,
    growth_factor: float = 0.45,
    fade: bool = True,
    fade_steps: Optional[int] = None,
    fade_alpha_min: float = 0.12,
) -> FuncAnimation:
    if len(walk) < 2:
        raise ValueError("walk must contain at least two vertices")

    pos2d = _compute_pos2d_from_graph(G)

    # Build list of sequential base segments (without offsets)
    segments_base: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(walk) - 1):
        u, v = walk[i], walk[i + 1]
        segments_base.append((pos2d[u], pos2d[v]))

    n_steps = len(segments_base)
    cmap = cm.get_cmap(cmap_name)
    base_colors = [cmap(i / max(1, n_steps - 1)) for i in range(n_steps)]  # RGBA tuples

    # Figure / axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw base faint graph (nodes/edges)
    nx.draw_networkx_nodes(G, pos2d, node_size=node_size, node_color="lightgray",
                           edgecolors="k", linewidths=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos2d, font_size=9, ax=ax)
    weights = nx.get_edge_attributes(G, "weight")
    if weights:
        max_w = max(abs(w) for w in weights.values()) if weights else 1.0
        lw_list = [max(0.5, 2.0 * (weights.get((u, v), weights.get((v, u), 1.0)) / max_w))
                   for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos2d, width=cast(Any, lw_list), alpha=0.25, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos2d, alpha=0.25, ax=ax)
    if show_edge_weights and weights:
        edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in weights.items()}
        nx.draw_networkx_edge_labels(G, pos2d, edge_labels=edge_labels, font_size=7, ax=ax)

    # LineCollection for trail — do NOT set a global alpha here
    trail_lc = LineCollection([], linewidths=repeat_linewidth_base, zorder=4)
    ax.add_collection(trail_lc)

    # moving marker for current node
    current_scatter = ax.scatter([], [], s=node_size * 0.6, c=[(0, 0, 0, 1.0)], zorder=6)

    arrow_patches: List[Any] = []
    annotation_texts: List[Any] = []

    # helper: compute offset for repeated traversals (hybrid increasing offset)
    def compute_offset_for_edge(u: int, v: int, occurrence_index: int, delta: float, growth: float):
        (x1, y1), (x2, y2) = pos2d[u], pos2d[v]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return (0.0, 0.0)
        ux = -dy / length
        uy = dx / length
        side = 1 if (occurrence_index % 2 == 0) else -1
        layer_base = 1.0 + (occurrence_index // 2) * growth
        offset_amount = delta * side * layer_base
        return (ux * offset_amount, uy * offset_amount)

    # compute span and base delta
    xs = [p[0] for p in pos2d.values()]
    ys = [p[1] for p in pos2d.values()]
    span = max(1e-6, max(max(xs) - min(xs), max(ys) - min(ys)))
    delta = repeat_offset_scale * span

    # precompute frozenset edge ids for each step
    edge_ids_per_step = [frozenset({walk[i], walk[i+1]}) for i in range(len(walk) - 1)]

    steps_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                         fontsize=9, bbox=dict(facecolor='white', alpha=0.8), zorder=7)

    def make_arrows_upto(k: int) -> List[FancyArrowPatch]:
        patches: List[FancyArrowPatch] = []
        for idx in range(0, min(k, n_steps), arrow_every):
            (x1, y1), (x2, y2) = segments_base[idx]
            dx = x2 - x1; dy = y2 - y1
            fx = x1 + 0.55 * dx; fy = y1 + 0.55 * dy
            p = FancyArrowPatch((fx, fy), (fx + 1e-9 + 0.001 * dx, fy + 1e-9 + 0.001 * dy),
                                arrowstyle='-|>', mutation_scale=10, color=base_colors[idx],
                                linewidth=0.6, alpha=0.95, zorder=5)
            patches.append(p)
        return patches

    # animation update
    def update(frame: int):
        k = frame
        # counts so far
        traversal_counts: Dict[frozenset, int] = {}
        occurrence_index_per_step: List[int] = []
        for i in range(k):
            eid = edge_ids_per_step[i]
            cnt = traversal_counts.get(eid, 0)
            occurrence_index_per_step.append(cnt)
            traversal_counts[eid] = cnt + 1

        adjusted_segments = []
        linewidths = []
        segment_rgba_colors = []

        # determine fade window
        effective_fade_steps = fade_steps if (fade_steps is not None and fade_steps > 0) else n_steps

        for i in range(k):
            (x1, y1), (x2, y2) = segments_base[i]
            occ_idx = occurrence_index_per_step[i]
            ox, oy = compute_offset_for_edge(walk[i], walk[i+1], occ_idx, delta, growth_factor)
            seg = ((x1 + ox, y1 + oy), (x2 + ox, y2 + oy))
            adjusted_segments.append(seg)

            lw = repeat_linewidth_base + repeat_linewidth_step * occ_idx
            linewidths.append(lw)

            # base color (r,g,b,a)
            r, g, b, _ = base_colors[i]

            # compute alpha via fading (newest segment should be alpha ~1.0)
            if fade:
                age = (k - 1) - i
                age = max(0, age)
                normalized = min(1.0, age / max(1, effective_fade_steps - 1))
                alpha = fade_alpha_min + (1.0 - fade_alpha_min) * (1.0 - normalized)
            else:
                alpha = 1.0

            segment_rgba_colors.append((r, g, b, alpha))

        # update LineCollection — use explicit setters that Matplotlib honors reliably
        trail_lc.set_segments(adjusted_segments)
        if adjusted_segments:
            trail_lc.set_linewidth(linewidths)
            trail_lc.set_colors(segment_rgba_colors)   # IMPORTANT: use set_colors for RGBA list
        else:
            trail_lc.set_colors([])

        # current marker
        if k == 0:
            cur_node = walk[0]
        else:
            cur_node = walk[k]
        cx, cy = pos2d[cur_node]
        current_scatter.set_offsets([[cx, cy]])

        # update annotations
        nonlocal annotation_texts
        for t in annotation_texts:
            try:
                t.remove()
            except Exception:
                pass
        annotation_texts = []

        if annotate_nodes_during_play:
            visits_so_far: Dict[int, int] = {}
            annotations: List[Tuple[int, int]] = []
            for idx in range(k + 1):
                node = walk[idx]
                if annotate_first_visit_only:
                    if node not in visits_so_far:
                        visits_so_far[node] = idx
                        annotations.append((node, idx))
                else:
                    annotations.append((node, idx))
            for node, idx in annotations:
                x, y = pos2d[node]
                dx = 0.012 * (1 if x >= 0 else -1)
                dy = 0.012 * (1 if y >= 0 else -1)
                t = ax.text(x + dx, y + dy, str(idx), fontsize=8, fontweight='bold',
                            color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5),
                            zorder=8)
                annotation_texts.append(t)

        # arrows update
        nonlocal arrow_patches
        for p in arrow_patches:
            try:
                p.remove()
            except Exception:
                pass
        arrow_patches = make_arrows_upto(k)
        for p in arrow_patches:
            ax.add_patch(p)

        # step text
        steps_text.set_text(f"step: {k}/{n_steps}")

        artists = [trail_lc, current_scatter, steps_text] + arrow_patches + annotation_texts
        return artists

    # axis limits
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = (xmax - xmin) * 0.06 if xmax != xmin else 0.1
    pady = (ymax - ymin) * 0.06 if ymax != ymin else 0.1
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)

    frames = list(range(0, n_steps + 1))
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=repeat)

    plt.show()
    return anim
