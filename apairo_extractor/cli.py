#!/usr/bin/env python3
"""Rosbag → KITTI/ZARR extractor with optional apairo preprocessing.

Usage:
  apairo-extract           # interactive TUI
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

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
from apairo_extractor.extract import extract_bag
from apairo_extractor.export_mnt import MntExportConfig, export_to_mnt, extract_bag_to_mnt
from apairo_extractor.preprocess import (
    PreprocessConfig,
    discover_preprocessors,
    run_preprocessors,
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


# ── Extraction runner ──────────────────────────────────────────────────────────


def _topic_for_channel(channel: str | None, topics: list[str]) -> str | None:
    """Reverse-map a KITTI channel dir name to its original ROS topic name."""
    if channel is None:
        return None
    for t in topics:
        if topic_to_dir(t) == channel:
            return t
    return None


def _run_extraction(
    bags: list[BagInfo],
    topics: list[str],
    preprocess_configs: list[PreprocessConfig],
    output_dir: Optional[Path],
    mnt_config: Optional[MntExportConfig] = None,
) -> None:
    mnt_only = output_dir is None
    if not mnt_only:
        output_dir.mkdir(parents=True, exist_ok=True)
    if mnt_config is not None:
        mnt_config.output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-resolve topic names for direct MNT extraction.
    mnt_points_topic  = _topic_for_channel(mnt_config.points_channel,   topics) if mnt_config else None
    mnt_image_topic   = _topic_for_channel(mnt_config.image_channel,    topics) if mnt_config else None
    mnt_odom_topic    = _topic_for_channel(mnt_config.odometry_channel,  topics) if mnt_config else None

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

        for bag in bags:
            mission_dir = mnt_config.output_dir / bag.path.stem if mnt_config else None

            if mnt_only:
                # Direct bag → MNT, no intermediate KITTI files.
                mnt_total = sum(t.msgcount for t in bag.topics if t.name in
                                {mnt_points_topic, mnt_image_topic, mnt_odom_topic} - {None})
                mnt_task = progress.add_task(
                    f"[blue]{bag.path.stem[:33]}", total=max(mnt_total, 1)
                )

                def _mnt_direct_cb(stage, done, total, _t=mnt_task):
                    progress.update(_t, completed=done, total=max(total, 1))

                extract_bag_to_mnt(
                    bag, mission_dir,
                    points_topic=mnt_points_topic,
                    image_topic=mnt_image_topic,
                    odometry_topic=mnt_odom_topic,
                    max_points=mnt_config.max_points,
                    progress_cb=_mnt_direct_cb,
                )
                progress.update(mnt_task, completed=max(mnt_total, 1))

            else:
                # KITTI extraction (+ optional MNT conversion from KITTI).
                topic_total = sum(t.msgcount for t in bag.topics if t.name in set(topics))
                bag_task = progress.add_task(
                    f"[green]{bag.path.stem[:33]}", total=max(topic_total, 1)
                )

                def _cb(done, total, _t=bag_task):
                    progress.update(_t, completed=done, total=max(total, 1))

                seq_dir, skipped = extract_bag(bag, topics, output_dir, progress_cb=_cb)
                progress.update(bag_task, completed=max(topic_total, 1))

                for s in skipped:
                    all_skipped.append((bag.path.name, s))

                if preprocess_configs:
                    pp_task = progress.add_task(
                        f"[magenta]preprocess {bag.path.stem[:27]}", total=1
                    )
                    run_preprocessors(seq_dir, preprocess_configs)
                    progress.update(pp_task, completed=1)

                if mnt_config is not None:
                    mnt_task = progress.add_task(
                        f"[blue]MNT {bag.path.stem[:30]}", total=3
                    )

                    def _mnt_cb(stage, done, total, _t=mnt_task):
                        progress.update(_t, description=f"[blue]MNT {stage:<10}", advance=0)
                        if done >= total:
                            progress.advance(_t, 1)

                    export_to_mnt(seq_dir, mission_dir, mnt_config, progress_cb=_mnt_cb)
                    progress.update(mnt_task, completed=3)

            progress.advance(overall, 1)

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
            state = "CONFIRM" if mnt_only else "OUTPUT_DIR"

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
            state = "CONFIRM"

        # ── CONFIRM ───────────────────────────────────────────────────────────
        elif state == "CONFIRM":
            console.print()
            console.print(_summary_panel(selected_bags, selected_topics, preprocess_configs, output_dir, mnt_config))
            console.print()

            action = questionary.select(
                "Ready?",
                choices=[
                    questionary.Choice("✓  Start extraction",             value="go"),
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
                console.print()
                _run_extraction(selected_bags, selected_topics, preprocess_configs, output_dir, mnt_config)
                return


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        interactive()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
