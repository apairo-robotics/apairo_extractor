#!/usr/bin/env python3
"""Rosbag → KITTI/ZARR extractor with optional apairo preprocessing.

Usage:
  apairo-extractor                                   # interactive TUI
  apairo-extractor --input DIR --list               # discover bags/topics
  apairo-extractor -i DIR -t /lidar /imu -o OUT      # headless extraction
  apairo-extractor -i DIR -t /lidar -o OUT -w 4      # … with 4 parallel workers
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box
from rich.text import Text

from apairo_extractor.bag import (
    BagInfo,
    read_bag_info,
    find_bags,
    topic_to_dir,
    compute_topic_coverage,
    get_topic_info,
    topics_in_bag,
)
from apairo_extractor.export_mnt import MntExportConfig
from apairo_extractor.preprocess import PreprocessConfig, discover_preprocessors
from apairo_extractor.runner import resolve_workers, run_extraction
from apairo_extractor.resources import (
    SystemResources,
    estimate_output_mb,
    estimate_worker_ram_mb,
    projected_ram_mb,
    read_resources,
    recommend_workers,
    worker_fit,
)

console = Console()

TUI_STYLE = Style([
    ("qmark",       "fg:#ff9d00 bold"),
    ("question",    "bold"),
    ("answer",      "fg:#00e5ff bold"),
    ("pointer",     "fg:#ff9d00 bold"),
    ("highlighted", "fg:#ff9d00 bold"),
    ("selected",    "fg:#00e5ff"),
    ("separator",   "fg:#555555"),
    ("instruction", "fg:#555555 italic"),
])

BANNER = """\
    _    ____   _    ___ ____   ___
   / \\  |  _ \\ / \\  |_ _|  _ \\ / _ \\
  / _ \\ | |_) / _ \\  | || |_) | | | |
 / ___ \\|  __/ ___ \\ | ||  _ <| |_| |
/_/   \\_\\_| /_/   \\_\\___|_| \\_\\\\___/

             E X T R A C T

          → KITTI • ZARR
"""
# ── Display helpers ────────────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB"


def _fmt_dur(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def _print_header() -> None:
    console.print(Panel(Text(BANNER, style="bold cyan"), border_style="dim cyan", padding=(0, 2)))
    console.print()


def _bags_table(bags: list[BagInfo]) -> Table:
    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="bold cyan",
              show_edge=False, title="[bold]Available bags[/bold]", title_justify="left")
    t.add_column("Name",       style="white")
    t.add_column("Size",       justify="right", style="yellow")
    t.add_column("Duration",   justify="right", style="green")
    t.add_column("Messages",   justify="right", style="dim")
    t.add_column("Topics",     justify="right", style="dim")
    for b in bags:
        t.add_row(
            b.path.name,
            _fmt_size(b.size_bytes),
            _fmt_dur(b.duration_s),
            f"{b.message_count:,}",
            str(len(b.topics)),
        )
    return t


def _topics_table(topics: list[str], bags: list[BagInfo], partial: list[str]) -> Table:
    partial_set = set(partial)
    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="bold cyan",
              show_edge=False, title="[bold]Topics[/bold]", title_justify="left")
    t.add_column("Topic",    style="white")
    t.add_column("Type",     style="dim")
    t.add_column("Messages", justify="right", style="dim")
    t.add_column("Coverage", style="green")
    for name in topics:
        info = get_topic_info(bags, name)
        msgtype = (info.msgtype or "?").split("/")[-1] if info else "?"
        count = f"{info.msgcount:,}" if info else "?"
        n_bags = sum(1 for b in bags if name in topics_in_bag(b))
        n_total = len(bags)
        if name in partial_set:
            cov = f"[yellow]{n_bags}/{n_total} bags ⚠[/yellow]"
        else:
            cov = f"[green]{n_bags}/{n_total} bags[/green]"
        t.add_row(name, msgtype, count, cov)
    return t


def _preprocess_table(preprocessors: dict[str, type]) -> Table:
    from apairo import FramePreprocessor

    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="bold cyan",
              show_edge=False, title="[bold]Discovered preprocessors[/bold]", title_justify="left")
    t.add_column("Name",        style="white")
    t.add_column("Kind",        style="dim")
    t.add_column("output_key",  style="cyan")
    t.add_column("input_keys",  style="dim")
    for name, cls in preprocessors.items():
        kind = "frame" if issubclass(cls, FramePreprocessor) else "sequence"
        out_key = getattr(cls, "output_key", "?")
        in_keys = ", ".join(getattr(cls, "input_keys", []))
        t.add_row(name, kind, out_key, in_keys)
    return t


def _summary_panel(
    bags: list[BagInfo],
    topics: list[str],
    preprocess_configs: list[PreprocessConfig],
    output_dir: Path,
    mnt_config: Optional[MntExportConfig],
) -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("k", style="bold dim", no_wrap=True)
    t.add_column("v", style="white")

    bag_str = ", ".join(b.path.name for b in bags[:3])
    if len(bags) > 3:
        bag_str += f" … +{len(bags) - 3}"
    t.add_row("Bags",    f"{len(bags)}  [{bag_str}]")

    ch_str = ", ".join(topic_to_dir(tp) for tp in topics[:4])
    if len(topics) > 4:
        ch_str += f" … +{len(topics) - 4}"
    t.add_row("Topics",  f"{len(topics)}  [{ch_str}]")
    t.add_row("Output",  str(output_dir))

    if preprocess_configs:
        pp = " | ".join(
            f"{c.output_key} ← {list(c.key_map.values())}"
            for c in preprocess_configs
        )
        t.add_row("Preprocess", pp)
    else:
        t.add_row("Preprocess", "[dim]none[/dim]")

    if mnt_config is not None:
        parts = []
        if mnt_config.points_channel:
            parts.append(f"points ← {mnt_config.points_channel}")
        if mnt_config.image_channel:
            parts.append(f"image ← {mnt_config.image_channel}")
        if mnt_config.odometry_channel:
            parts.append(f"odom ← {mnt_config.odometry_channel}")
        t.add_row("MNT export", (", ".join(parts) or "[dim]no channels mapped[/dim]")
                  + f"\n  → {mnt_config.output_dir}")
    else:
        t.add_row("MNT export", "[dim]none[/dim]")

    return Panel(t, title="[bold]Extraction summary[/bold]", border_style="yellow", padding=(0, 1))


# ── Resource estimation ───────────────────────────────────────────────────────


def _fmt_gb(mb: int) -> str:
    return f"{mb / 1024:.1f} GB"


_FIT_STYLE = {"ok": ("green", "✓"), "warn": ("yellow", "⚠"), "bad": ("red", "✗")}


def _worker_choices(res: SystemResources, n_bags: int, recommended: int) -> list[int]:
    """Candidate worker counts: 1, 2, 4, recommended, and the practical max."""
    cands = {1, 2, 4, recommended, min(res.cpu_count, max(1, n_bags))}
    return sorted(c for c in cands if 1 <= c <= max(1, n_bags))


def _resources_panel(
    res: SystemResources, n_bags: int, input_mb: int, output_mb: int,
    recommended: int, per_worker_mb: int,
) -> Panel:
    """Show what the machine has, what's in use, and the cost of N workers."""
    sys_t = Table(box=None, show_header=False, padding=(0, 2))
    sys_t.add_column("k", style="bold dim", no_wrap=True)
    sys_t.add_column("v", style="white")
    sys_t.add_row("CPU", f"{res.cpu_count} cores · load {res.load_avg_1m:.2f}")
    sys_t.add_row(
        "RAM",
        f"{_fmt_gb(res.ram_available_mb)} free / {_fmt_gb(res.ram_total_mb)} "
        f"([{'red' if res.ram_used_pct > 85 else 'dim'}]{res.ram_used_pct:.0f}% used[/])",
    )
    disk_low = res.disk_free_mb and res.disk_free_mb < output_mb
    disk_tag = {"HDD": " [yellow]· HDD (rotational)[/yellow]",
                "SSD/NVMe": " [dim]· SSD/NVMe[/dim]"}.get(res.disk_kind, "")
    sys_t.add_row(
        "Disk (output)",
        f"[{'red' if disk_low else 'white'}]{_fmt_gb(res.disk_free_mb)} free[/] "
        f"/ {_fmt_gb(res.disk_total_mb)}{disk_tag}",
    )
    sys_t.add_row("Input selected", f"{_fmt_gb(input_mb)} across {n_bags} bag(s)")
    sys_t.add_row(
        "Est. output",
        f"[{'red' if disk_low else 'white'}]~{_fmt_gb(output_mb)}[/] (selected channels)",
    )
    sys_t.add_row("RAM / worker", f"~{_fmt_gb(per_worker_mb)} (est.)")

    cmp_t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="bold cyan",
                  show_edge=False, padding=(0, 2))
    cmp_t.add_column("Workers", justify="right")
    cmp_t.add_column("CPU use")
    cmp_t.add_column("RAM (est.)", justify="right")
    cmp_t.add_column("Fit")
    for w in _worker_choices(res, n_bags, recommended):
        kind, note = worker_fit(w, res, per_worker_mb)
        colour, sym = _FIT_STYLE[kind]
        tag = "  ← recommended" if w == recommended else ""
        label = f"{w}{tag}"
        fit = f"[{colour}]{sym}[/]" + (f" [dim]{note}[/dim]" if note else "")
        cmp_t.add_row(
            label,
            f"{w}/{res.cpu_count} cores",
            _fmt_gb(projected_ram_mb(w, per_worker_mb)),
            fit,
        )

    body = Table.grid(padding=(1, 0))
    body.add_row(sys_t)
    body.add_row(cmp_t)
    if res.disk_kind == "HDD":
        body.add_row(
            "[yellow]Rotational disk: parallel writes thrash the head — "
            f"keeping ≤{recommended} worker(s) recommended.[/yellow]"
        )
    if n_bags <= 1:
        body.add_row("[dim]Only one bag selected — parallelism across bags does not apply.[/dim]")
    return Panel(body, title="[bold]Resources & parallelism[/bold]",
                 border_style="cyan", padding=(0, 1))


# ── Preprocessor configuration wizard ─────────────────────────────────────────


def _configure_preprocessor(
    cls: type,
    cls_key: str,
    extracted_channels: list[str],
) -> Optional[PreprocessConfig]:
    """Interactively configure a single preprocessor. Returns None if cancelled."""
    default_out = getattr(cls, "output_key", cls.__name__.lower())
    default_inputs: list[str] = getattr(cls, "input_keys", [])

    console.print(f"\n  Configuring [bold cyan]{cls_key}[/bold cyan]")

    out_key = questionary.text(
        f"  Output channel name:",
        default=default_out,
        style=TUI_STYLE,
    ).ask()
    if out_key is None:
        return None
    out_key = out_key.strip() or default_out

    key_map: dict[str, str] = {}
    for preproc_input in default_inputs:
        if not extracted_channels:
            console.print(f"  [yellow]No extracted channels to map '{preproc_input}' to.[/yellow]")
            key_map[preproc_input] = preproc_input
            continue

        chosen = questionary.select(
            f"  Map preprocessor input [cyan]'{preproc_input}'[/cyan] to which extracted channel?",
            choices=extracted_channels,
            style=TUI_STYLE,
        ).ask()
        if chosen is None:
            return None
        key_map[preproc_input] = chosen

    return PreprocessConfig(cls=cls, output_key=out_key, key_map=key_map)


# ── Extraction runner (rich frontend over runner.run_extraction) ──────────────


def _run_extraction(
    bags: list[BagInfo],
    topics: list[str],
    preprocess_configs: list[PreprocessConfig],
    output_dir: Optional[Path],
    mnt_config: Optional[MntExportConfig] = None,
    workers: Optional[int] = None,
) -> None:
    """Run an extraction with a live rich progress display (the TUI/CLI front)."""
    mnt_only = output_dir is None
    all_skipped: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<35}"),
        BarColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task("[cyan]Overall", total=len(bags))

        # Per-bag bars are created lazily on first progress so we don't flood
        # the display with idle bars when there are many bags.
        bag_tasks: dict[str, int] = {}

        def on_progress(bag_id, phase, done, total):
            task = bag_tasks.get(bag_id)
            if task is None:
                task = progress.add_task(f"[green]{bag_id[:33]}", total=max(total, 1))
                bag_tasks[bag_id] = task
            progress.update(
                task,
                description=f"[green]{bag_id[:25]} · {phase}",
                completed=done,
                total=max(total, 1),
            )

        def on_bag_done(name, skipped, err):
            if err:
                console.print(f"[red]Error extracting {name}: {err}[/red]")
            for s in skipped:
                all_skipped.append((name, s))
            progress.advance(overall, 1)

        run_extraction(
            bags, topics, output_dir,
            mnt_config=mnt_config,
            preprocess_configs=preprocess_configs,
            workers=workers,
            on_progress=on_progress,
            on_bag_done=on_bag_done,
        )

    console.print()
    if not mnt_only:
        console.print(
            f"[bold green]Done![/bold green]  "
            f"{len(bags)} sequence(s) → [cyan]{output_dir}[/cyan]"
        )
    if mnt_config is not None:
        console.print(
            f"[bold green]Done![/bold green]  "
            f"{len(bags)} mission(s) → [cyan]{mnt_config.output_dir}[/cyan]"
        )

    if all_skipped:
        console.print()
        console.print("[yellow]Skipped (missing or unsupported):[/yellow]")
        for bag_name, topic in all_skipped:
            console.print(f"  [dim]{bag_name}[/dim]  {topic}")


# ── Interactive TUI (state machine) ───────────────────────────────────────────


def interactive() -> None:
    _print_header()

    # Persistent state
    input_dir:         Optional[Path]             = None
    bag_paths:         list[Path]                 = []
    all_bags:          list[BagInfo]              = []
    selected_bags:     list[BagInfo]              = []
    selected_topics:   list[str]                  = []
    preprocess_configs: list[PreprocessConfig]    = []
    mnt_config:        Optional[MntExportConfig]  = None
    mnt_only:          bool                       = False
    output_dir:        Optional[Path]             = None
    workers:           Optional[int]              = None
    est_output_mb:     int                        = 0
    _found_preprocessors: dict[str, type]         = {}

    state = "INPUT_DIR"

    while True:

        # ── INPUT_DIR ─────────────────────────────────────────────────────────
        if state == "INPUT_DIR":
            raw = questionary.path(
                "Directory containing rosbags:",
                only_directories=True,
                style=TUI_STYLE,
            ).ask()
            if raw is None:
                console.print("[dim]Cancelled.[/dim]")
                return
            candidate = Path(os.path.expanduser(raw))
            if not candidate.is_dir():
                console.print(f"[red]Not a directory:[/red] {candidate}\n")
                continue
            with console.status("[dim]Scanning for .bag files…[/dim]"):
                bag_paths = find_bags(candidate)
            if not bag_paths:
                console.print(f"[yellow]No .bag files found in {candidate}[/yellow]\n")
                continue
            input_dir = candidate
            state = "BAG_LOAD"

        # ── BAG_LOAD ──────────────────────────────────────────────────────────
        elif state == "BAG_LOAD":
            errors: list[tuple[Path, Exception]] = []
            with console.status(f"[dim]Reading metadata for {len(bag_paths)} bag(s)…[/dim]"):
                all_bags = []
                for p in bag_paths:
                    try:
                        all_bags.append(read_bag_info(p))
                    except Exception as exc:
                        errors.append((p, exc))
            for p, exc in errors:
                console.print(f"  [red]Error reading {p.name}:[/red] {exc}")
            if not all_bags:
                console.print("[red]No readable bags found.[/red]\n")
                state = "INPUT_DIR"
                continue
            state = "BAG_SELECT"

        # ── BAG_SELECT ────────────────────────────────────────────────────────
        elif state == "BAG_SELECT":
            console.print()
            console.print(_bags_table(all_bags))
            console.print()

            prev_names = {b.path.name for b in selected_bags}
            choices = [
                questionary.Choice(b.path.name, value=b, checked=(b.path.name in prev_names))
                for b in all_bags
            ]
            sel = questionary.checkbox(
                "Select bags  (Space = toggle · a = all · Enter = confirm · Ctrl+C = back):",
                choices=choices,
                style=TUI_STYLE,
            ).ask()
            if sel is None:
                state = "INPUT_DIR"
                continue
            if not sel:
                console.print("[yellow]Select at least one bag.[/yellow]\n")
                continue
            selected_bags = sel
            state = "TOPIC_SELECT"

        # ── TOPIC_SELECT ──────────────────────────────────────────────────────
        elif state == "TOPIC_SELECT":
            console.print()
            common, partial = compute_topic_coverage(selected_bags)
            all_topics = common + partial

            if not all_topics:
                console.print("[yellow]No topics found in selected bags.[/yellow]\n")
                state = "BAG_SELECT"
                continue

            console.print(_topics_table(all_topics, selected_bags, partial))
            console.print()

            prev_sel = set(selected_topics)
            choices = [
                questionary.Choice(t, value=t, checked=(t in prev_sel))
                for t in common
            ]
            if partial:
                choices.append(questionary.Separator(f"── Partial — not in all bags ──"))
                for t in partial:
                    choices.append(
                        questionary.Choice(f"{t}  ⚠", value=t, checked=(t in prev_sel))
                    )

            sel = questionary.checkbox(
                "Select topics  (Space = toggle · a = all · Enter = confirm · Ctrl+C = back):",
                choices=choices,
                style=TUI_STYLE,
            ).ask()
            if sel is None:
                state = "BAG_SELECT"
                continue
            if not sel:
                console.print("[yellow]Select at least one topic.[/yellow]\n")
                continue

            # Warn about partial topics and offer to go back
            sel_partial = [t for t in sel if t in partial]
            if sel_partial:
                console.print()
                for topic in sel_partial:
                    missing_in = [
                        b.path.name
                        for b in selected_bags
                        if topic not in topics_in_bag(b)
                    ]
                    console.print(
                        f"  [yellow]⚠[/yellow] [bold]{topic}[/bold] absent from: "
                        + ", ".join(missing_in)
                    )
                console.print(
                    "\n  These topics will be [dim]skipped[/dim] for the bags that lack them.\n"
                )
                ok = questionary.confirm(
                    "Proceed with these partial topics?",
                    default=True,
                    style=TUI_STYLE,
                ).ask()
                if ok is None or not ok:
                    continue  # stay in TOPIC_SELECT

            selected_topics = sel
            state = "PREPROCESS_CHOICE"

        # ── PREPROCESS_CHOICE ─────────────────────────────────────────────────
        elif state == "PREPROCESS_CHOICE":
            console.print()
            action = questionary.select(
                "Preprocessing?",
                choices=[
                    questionary.Choice("None — skip preprocessing",    value="skip"),
                    questionary.Choice("Add an apairo preprocessing pipeline", value="add"),
                    questionary.Separator(),
                    questionary.Choice("← Back to topic selection",    value="back"),
                ],
                style=TUI_STYLE,
            ).ask()
            if action is None or action == "back":
                state = "TOPIC_SELECT"
            elif action == "skip":
                preprocess_configs = []
                state = "MNT_CHOICE"
            else:
                state = "PREPROCESS_DIR"

        # ── PREPROCESS_DIR ────────────────────────────────────────────────────
        elif state == "PREPROCESS_DIR":
            console.print()
            raw = questionary.path(
                "Directory containing preprocessor classes  (Ctrl+C = back):",
                only_directories=True,
                style=TUI_STYLE,
            ).ask()
            if raw is None:
                state = "PREPROCESS_CHOICE"
                continue
            preproc_dir = Path(os.path.expanduser(raw))
            if not preproc_dir.is_dir():
                console.print(f"[red]Not a directory:[/red] {preproc_dir}\n")
                continue
            with console.status("[dim]Discovering preprocessors…[/dim]"):
                found = discover_preprocessors(preproc_dir)
            if not found:
                console.print(
                    "[yellow]No FramePreprocessor / SequencePreprocessor subclasses found "
                    "in that directory.[/yellow]\n"
                )
                continue
            _found_preprocessors = found
            console.print()
            console.print(_preprocess_table(found))
            console.print()
            state = "PREPROCESS_SELECT"

        # ── PREPROCESS_SELECT ─────────────────────────────────────────────────
        elif state == "PREPROCESS_SELECT":
            sel = questionary.checkbox(
                "Select preprocessors to apply  (Space = toggle · Enter = confirm · Ctrl+C = back):",
                choices=list(_found_preprocessors.keys()),
                style=TUI_STYLE,
            ).ask()
            if sel is None:
                state = "PREPROCESS_DIR"
                continue
            if not sel:
                preprocess_configs = []
                state = "MNT_CHOICE"
                continue

            extracted_channels = [topic_to_dir(t) for t in selected_topics]
            configs: list[PreprocessConfig] = []
            cancelled = False
            for key in sel:
                cfg = _configure_preprocessor(
                    _found_preprocessors[key], key, extracted_channels
                )
                if cfg is None:
                    cancelled = True
                    break
                configs.append(cfg)

            if cancelled:
                state = "PREPROCESS_SELECT"
                continue

            preprocess_configs = configs
            state = "MNT_CHOICE"

        # ── MNT_CHOICE ────────────────────────────────────────────────────────
        elif state == "MNT_CHOICE":
            console.print()
            action = questionary.select(
                "Output format?",
                choices=[
                    questionary.Choice("KITTI only",       value="kitti"),
                    questionary.Choice("MNT only (zarr)",  value="mnt"),
                    questionary.Choice("Both",             value="both"),
                    questionary.Separator(),
                    questionary.Choice("← Back",           value="back"),
                ],
                style=TUI_STYLE,
            ).ask()
            if action is None or action == "back":
                state = "PREPROCESS_CHOICE"
            elif action == "kitti":
                mnt_config = None
                mnt_only = False
                state = "OUTPUT_DIR"
            elif action == "mnt":
                mnt_only = True
                state = "MNT_CONFIG"
            else:  # both
                mnt_only = False
                state = "MNT_CONFIG"

        # ── MNT_CONFIG ────────────────────────────────────────────────────────
        elif state == "MNT_CONFIG":
            extracted_channels = [topic_to_dir(t) for t in selected_topics]
            none_choice = "(none)"
            choices_with_none = [none_choice] + extracted_channels

            console.print()
            console.print("  Map extracted channels to MNT roles  (Enter to confirm · Ctrl+C = back)\n")

            prev = mnt_config or MntExportConfig()

            pts_ch = questionary.select(
                "  [points] PointCloud2 channel:",
                choices=choices_with_none,
                default=prev.points_channel or none_choice,
                style=TUI_STYLE,
            ).ask()
            if pts_ch is None:
                state = "MNT_CHOICE"
                continue

            img_ch = questionary.select(
                "  [image] Camera / image channel:",
                choices=choices_with_none,
                default=prev.image_channel or none_choice,
                style=TUI_STYLE,
            ).ask()
            if img_ch is None:
                state = "MNT_CHOICE"
                continue

            odom_ch = questionary.select(
                "  [odometry] Odometry / PoseStamped channel  (position + yaw):",
                choices=choices_with_none,
                default=prev.odometry_channel or none_choice,
                style=TUI_STYLE,
            ).ask()
            if odom_ch is None:
                state = "MNT_CHOICE"
                continue

            default_mnt_out = str(prev.output_dir) if prev.output_dir else str(Path.home() / "mnt_exported")
            raw_mnt = questionary.path(
                "  MNT output directory:",
                default=default_mnt_out,
                only_directories=True,
                style=TUI_STYLE,
            ).ask()
            if raw_mnt is None:
                state = "MNT_CHOICE"
                continue

            mnt_config = MntExportConfig(
                points_channel=pts_ch if pts_ch != none_choice else None,
                image_channel=img_ch if img_ch != none_choice else None,
                odometry_channel=odom_ch if odom_ch != none_choice else None,
                output_dir=Path(os.path.expanduser(raw_mnt)),
            )
            state = "WORKERS" if mnt_only else "OUTPUT_DIR"

        # ── OUTPUT_DIR ────────────────────────────────────────────────────────
        elif state == "OUTPUT_DIR":
            console.print()
            default = str(output_dir) if output_dir else str(Path.home() / "kitti_extracted")
            raw = questionary.path(
                "Output directory  (Ctrl+C = back):",
                default=default,
                only_directories=True,
                style=TUI_STYLE,
            ).ask()
            if raw is None:
                state = "MNT_CHOICE"
                continue
            output_dir = Path(os.path.expanduser(raw))
            state = "WORKERS"

        # ── WORKERS ───────────────────────────────────────────────────────────
        elif state == "WORKERS":
            console.print()
            n_bags = len(selected_bags)
            res_target = output_dir if not mnt_only else (
                mnt_config.output_dir if mnt_config else None
            )
            res = read_resources(res_target)
            input_mb = sum(b.size_bytes for b in selected_bags) // (1024 * 1024)
            per_worker_mb = estimate_worker_ram_mb(selected_bags, selected_topics)
            recommended = recommend_workers(n_bags, res, per_worker_mb)
            with console.status("[dim]Estimating output size…[/dim]"):
                est_output_mb = estimate_output_mb(selected_bags, selected_topics)

            console.print(_resources_panel(
                res, n_bags, input_mb, est_output_mb, recommended, per_worker_mb))
            console.print()

            if n_bags <= 1:
                # Nothing to parallelise across; skip the prompt.
                workers = 1
                state = "CONFIRM"
                continue

            candidates = _worker_choices(res, n_bags, recommended)
            current = workers if workers in candidates else recommended
            choices = []
            for w in candidates:
                kind, note = worker_fit(w, res, per_worker_mb)
                _, sym = _FIT_STYLE[kind]
                tag = "  (recommended)" if w == recommended else ""
                suffix = f"  {sym} {note}" if note else f"  {sym}"
                choices.append(questionary.Choice(
                    f"{w} worker(s) · ~{_fmt_gb(projected_ram_mb(w, per_worker_mb))} RAM{tag}{suffix}",
                    value=w,
                    checked=(w == current),
                ))
            choices.append(questionary.Separator())
            choices.append(questionary.Choice("Custom…", value="custom"))
            choices.append(questionary.Choice("← Back", value="back"))

            sel = questionary.select(
                "Parallel workers  (bags extracted at once):",
                choices=choices,
                default=current,
                style=TUI_STYLE,
            ).ask()
            if sel is None or sel == "back":
                state = "OUTPUT_DIR" if not mnt_only else "MNT_CONFIG"
                continue
            if sel == "custom":
                raw = questionary.text(
                    f"Number of workers (1–{n_bags}):",
                    default=str(recommended),
                    style=TUI_STYLE,
                ).ask()
                if raw is None:
                    continue  # stay in WORKERS
                try:
                    sel = max(1, min(int(raw), n_bags))
                except ValueError:
                    console.print("[red]Enter a whole number.[/red]\n")
                    continue
            workers = sel
            state = "CONFIRM"

        # ── CONFIRM ───────────────────────────────────────────────────────────
        elif state == "CONFIRM":
            console.print()
            console.print(_summary_panel(selected_bags, selected_topics, preprocess_configs, output_dir, mnt_config))
            console.print()

            workers_label = (
                f"✓  Start extraction  [{workers} worker(s)]"
                if workers and len(selected_bags) > 1
                else "✓  Start extraction"
            )
            action = questionary.select(
                "Ready?",
                choices=[
                    questionary.Choice(workers_label,                     value="go"),
                    questionary.Choice("←  Change parallel workers",      value="workers"),
                    questionary.Choice("←  Change output directory",      value="output"),
                    questionary.Choice("←  Change MNT export",            value="mnt"),
                    questionary.Choice("←  Change preprocessing",         value="preprocess"),
                    questionary.Choice("←  Change topics",                value="topics"),
                    questionary.Choice("←  Change bags",                  value="bags"),
                    questionary.Choice("✕  Cancel",                       value="cancel"),
                ],
                style=TUI_STYLE,
            ).ask()

            if action is None or action == "cancel":
                console.print("[dim]Cancelled.[/dim]")
                return
            elif action == "workers":
                state = "WORKERS"
            elif action == "output":
                state = "OUTPUT_DIR"
            elif action == "mnt":
                state = "MNT_CHOICE"
            elif action == "preprocess":
                state = "PREPROCESS_CHOICE"
            elif action == "topics":
                state = "TOPIC_SELECT"
            elif action == "bags":
                state = "BAG_SELECT"
            else:
                # Soft disk-space gate against the estimated output size for the
                # selected channels (not the whole bag) — warn, but let the user
                # proceed, when free space is below it.
                out_target = output_dir if not mnt_only else (
                    mnt_config.output_dir if mnt_config else None
                )
                res = read_resources(out_target)
                if res.disk_free_mb and res.disk_free_mb < est_output_mb:
                    console.print(
                        f"[yellow]⚠ Only {_fmt_gb(res.disk_free_mb)} free on the output "
                        f"disk for ~{_fmt_gb(est_output_mb)} of estimated output "
                        f"(selected channels). It may not fit.[/yellow]"
                    )
                    proceed = questionary.confirm(
                        "Continue anyway?", default=False, style=TUI_STYLE,
                    ).ask()
                    if not proceed:
                        continue  # stay on CONFIRM
                console.print()
                _run_extraction(selected_bags, selected_topics, preprocess_configs,
                                output_dir, mnt_config, workers=workers)
                return


# ── Headless CLI ──────────────────────────────────────────────────────────────


def _load_bags(input_dir: Path, names: Optional[list[str]]) -> list[BagInfo]:
    """Discover and read bags under ``input_dir``, optionally filtered by name."""
    paths = find_bags(input_dir)
    if names:
        wanted = set(names)
        paths = [p for p in paths if p.name in wanted or p.stem in wanted]
    bags: list[BagInfo] = []
    for p in paths:
        try:
            bags.append(read_bag_info(p))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error reading {p.name}:[/red] {exc}")
    return bags


def _fail(msg: str) -> NoReturn:
    """Print a usage error and exit with code 2 (consistent with argparse)."""
    console.print(f"[red]{msg}[/red]")
    raise SystemExit(2)


def _parse_preprocess_spec(
    spec: str, found: dict[str, type], channels: list[str]
) -> PreprocessConfig:
    """Build a PreprocessConfig from a CLI spec string.

    SPEC = ``file.Class`` or ``file.Class:output=name,inputkey=channel,...``.
    Unspecified input keys default to a channel of the same name. Raises
    SystemExit with a helpful message on any error.
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if name not in found:
        avail = ", ".join(found) or "(none)"
        _fail(f"Unknown preprocessor '{name}'. Available: {avail}")
    cls = found[name]
    output_key = getattr(cls, "output_key", cls.__name__.lower())
    key_map = {k: k for k in getattr(cls, "input_keys", [])}

    for pair in (p for p in rest.split(",") if p.strip()):
        key, sep, val = pair.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not val:
            _fail(f"Bad mapping '{pair}' in --preprocess '{spec}' (expected key=value)")
        if key == "output":
            output_key = val
        else:
            key_map[key] = val

    for k, ch in key_map.items():
        if channels and ch not in channels:
            console.print(
                f"[yellow]⚠ preprocessor '{name}': input '{k}' → channel '{ch}' "
                f"is not among the extracted channels ({', '.join(channels)}).[/yellow]"
            )
    return PreprocessConfig(cls=cls, output_key=output_key, key_map=key_map)


def _build_preprocess_configs(
    args: argparse.Namespace, channels: list[str]
) -> list[PreprocessConfig]:
    """Resolve --preprocess specs against --preprocess-dir (headless)."""
    if not args.preprocess:
        return []
    if not args.preprocess_dir:
        _fail("--preprocess requires --preprocess-dir")
    pp_dir = Path(os.path.expanduser(args.preprocess_dir))
    if not pp_dir.is_dir():
        _fail(f"--preprocess-dir is not a directory: {pp_dir}")
    found = discover_preprocessors(pp_dir)
    if not found:
        _fail(f"No preprocessor classes found in {pp_dir}")
    return [_parse_preprocess_spec(spec, found, channels) for spec in args.preprocess]


def _run_headless(args: argparse.Namespace) -> int:
    """Non-interactive extraction driven entirely by CLI flags."""
    input_dir = Path(os.path.expanduser(args.input))
    if not input_dir.is_dir():
        console.print(f"[red]Not a directory:[/red] {input_dir}")
        return 2

    with console.status(f"[dim]Reading bags in {input_dir}…[/dim]"):
        bags = _load_bags(input_dir, args.bags)
    if not bags:
        console.print("[red]No readable bags found.[/red]")
        return 1

    if args.list:
        common, partial = compute_topic_coverage(bags)
        console.print(_bags_table(bags))
        console.print()
        console.print(_topics_table(common + partial, bags, partial))
        return 0

    mnt_config = None
    if args.mnt_output or args.points or args.image or args.odom:
        mnt_config = MntExportConfig(
            points_channel=args.points,
            image_channel=args.image,
            odometry_channel=args.odom,
            output_dir=Path(os.path.expanduser(args.mnt_output)) if args.mnt_output
            else Path.home() / "mnt_exported",
        )
    output_dir = Path(os.path.expanduser(args.output)) if args.output else None
    mnt_only = output_dir is None and mnt_config is not None

    if not args.topics:
        console.print("[red]--topics is required (or use --list to discover them).[/red]")
        return 2
    if output_dir is None and mnt_config is None:
        console.print("[red]Provide --output (KITTI) and/or --mnt-output (MNT).[/red]")
        return 2

    available = set().union(*(topics_in_bag(b) for b in bags))
    unknown = [t for t in args.topics if t not in available]
    if unknown:
        console.print(f"[yellow]Topics not present in any bag (will be skipped):[/yellow] {', '.join(unknown)}")

    channels = [topic_to_dir(t) for t in args.topics]
    preprocess_configs = _build_preprocess_configs(args, channels)

    # Soft disk-space warning against the estimated output for the chosen topics.
    res = read_resources(output_dir if output_dir is not None else
                         (mnt_config.output_dir if mnt_config else None))
    est_output_mb = estimate_output_mb(bags, args.topics)
    if res.disk_free_mb and res.disk_free_mb < est_output_mb:
        console.print(
            f"[yellow]⚠ Only {_fmt_gb(res.disk_free_mb)} free for ~{_fmt_gb(est_output_mb)} "
            f"of estimated output (selected channels). It may not fit.[/yellow]"
        )

    console.print(
        f"[cyan]Extracting[/cyan] {len(bags)} bag(s) · {len(args.topics)} topic(s)"
        f"{f' · {len(preprocess_configs)} preprocessor(s)' if preprocess_configs else ''}"
        f" · workers={args.workers or resolve_workers(len(bags))}"
    )
    _run_extraction(bags, args.topics, preprocess_configs, output_dir, mnt_config,
                    workers=args.workers)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apairo-extractor",
        description="Extract ROS bags to KITTI/MNT. Run with no arguments for the "
                    "interactive TUI, or pass --input for headless extraction.",
    )
    p.add_argument("-i", "--input", help="Directory containing rosbags (enables headless mode)")
    p.add_argument("-o", "--output", help="KITTI output directory")
    p.add_argument("-t", "--topics", nargs="+", metavar="TOPIC", help="ROS topics to extract")
    p.add_argument("-w", "--workers", type=int, help="Parallel workers (default: auto)")
    p.add_argument("--bags", nargs="+", metavar="NAME", help="Only these bag names/stems (default: all found)")
    p.add_argument("-l", "--list", action="store_true", help="List bags and topics, then exit")
    p.add_argument("--preprocess-dir", help="Directory containing apairo preprocessor classes")
    p.add_argument(
        "--preprocess", action="append", metavar="SPEC", default=None,
        help="Apply a preprocessor (repeatable). SPEC = 'file.Class' or "
             "'file.Class:output=name,inputkey=channel,...' "
             "(unset inputs default to a channel of the same name).",
    )
    p.add_argument("--mnt-output", help="MNT/zarr output directory (enables MNT export)")
    p.add_argument("--points", help="KITTI channel → MNT points role")
    p.add_argument("--image", help="KITTI channel → MNT image role")
    p.add_argument("--odom", help="KITTI channel → MNT odometry role")
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    return p


def _version() -> str:
    from apairo_extractor import __version__
    return __version__


# ── Entry point ────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        if args.input is not None:
            sys.exit(_run_headless(args))
        interactive()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
