#!/usr/bin/env python3
"""
Climate sonification v2: NASA GISTEMP + Mauna Loa CO2 -> ABC / MIDI / web player.

Mapping
-------
temperature : monthly anomaly -> A minor pentatonic, A3..A5
rhythm      : month-to-month volatility -> note length / 16th-note flurries
articulation: month-to-month change > 0.15 -> staccato (55% length, crisp attack)
tempo       : decadal mean volatility -> 68..112 dotted-quarter bpm
bass        : syncopated root/fifth/octave pattern, root cycles A-C-D-E-G
CO2         : annual mean -> D Dorian counter-line, D3..D4, enters 1958
contrast    : NH-SH -> A minor pentatonic A5..A6, one sustained note per year
form        : 10-bar phrases, breath rests, cadence + 3-note decade motif

Usage: python3 sonify.py [--start 1925] [--end 2024] [--out warming]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mido

HERE = Path(__file__).resolve().parent
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# --- temperature voice: A minor pentatonic A C D E G, A3..A5 -------------
PENTA = [0, 3, 5, 7, 10]
ROOT_MIDI = 57  # A3
N_DEGREES = 10  # A3 .. A5
T_MIN, T_MAX = -0.6, 1.5

# --- CO2 voice: D Dorian D E F G A B C D, D3..D4 -------------------------
CO2_ROOT = 50  # D3
CO2_INTERVALS = [0, 2, 3, 5, 7, 9, 10, 12]
CO2_MIN, CO2_MAX = 315.0, 425.0

# --- hemispheric contrast voice: NH - SH -> A minor pentatonic, A5..A6 ---
CONTRAST_ROOT = 81  # A5
CONTRAST_MIN, CONTRAST_MAX = -0.4, 0.8

BAR_TICKS = 2880  # 12/8 at 480 ticks/quarter
EIGHTH = 240
TPQ = 480

VOICE_NAMES = {0: "melody", 1: "bass", 2: "co2", 3: "motif", 4: "contrast"}


@dataclass
class Note:
    tick: int
    dur: int
    midi: int
    vel: int
    voice: int
    art: int = 0  # articulation: -1 staccato, 0 normal (climate velocity -> attack)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_gistemp(path: Path):
    with path.open(newline="") as f:
        raw = list(csv.reader(f))
    start = next(i for i, r in enumerate(raw) if r and r[0].strip() == "Year")
    out = []
    for r in raw[start + 1 :]:
        if not r or not r[0].strip().isdigit():
            continue
        year = int(r[0])
        months = []
        for x in r[1:13]:
            x = x.strip()
            months.append(float(x) if x not in ("***", "") else None)
        if all(v is not None for v in months):
            out.append((year, months))
    return out


def load_co2(path: Path):
    """Annual mean of NOAA's deseasonalized monthly Mauna Loa CO2."""
    with path.open() as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("year,"))
    months = {}
    for line in lines[start + 1 :]:
        r = next(csv.reader([line]))
        if len(r) < 5 or not r[0].strip().isdigit():
            continue
        y, m = int(r[0]), int(r[1])
        try:
            val = float(r[4])  # deseasonalized
        except ValueError:
            continue
        months.setdefault(y, []).append(val)
    return {y: sum(v) / len(v) for y, v in months.items() if v}


def load_contrast(nh_path: Path, sh_path: Path, start_year: int, end_year: int):
    """Annual NH minus SH anomaly contrast; {} if the hemispheric tables are absent."""
    if not nh_path.exists() or not sh_path.exists():
        return {}
    nh = dict(load_gistemp(nh_path))
    sh = dict(load_gistemp(sh_path))
    out = {}
    for y in range(start_year, end_year + 1):
        if y in nh and y in sh:
            out[y] = sum(a - b for a, b in zip(nh[y], sh[y])) / 12
    return out


def degree_for(t: float) -> int:
    x = (t - T_MIN) / (T_MAX - T_MIN)
    return max(0, min(N_DEGREES, round(x * N_DEGREES)))


def midi_for(degree: int) -> int:
    return ROOT_MIDI + 12 * (degree // 5) + PENTA[degree % 5]


def co2_degree(v: float) -> int:
    x = (v - CO2_MIN) / (CO2_MAX - CO2_MIN)
    return max(0, min(len(CO2_INTERVALS) - 1, round(x * (len(CO2_INTERVALS) - 1))))


def co2_midi(degree: int) -> int:
    return CO2_ROOT + CO2_INTERVALS[degree]


def contrast_degree(v: float) -> int:
    x = (v - CONTRAST_MIN) / (CONTRAST_MAX - CONTRAST_MIN)
    return max(0, min(N_DEGREES, round(x * N_DEGREES)))


def contrast_midi(degree: int) -> int:
    return CONTRAST_ROOT + 12 * (degree // 5) + PENTA[degree % 5]


def abc_pitch(midi: int) -> str:
    # naturals used by A minor / D Dorian
    letter = {9: "A", 11: "B", 0: "C", 2: "D", 4: "E", 5: "F", 7: "G"}[midi % 12]
    octave = midi // 12 - 1
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter + "," * max(0, 4 - octave)


def abc_len(dur_eighths: float) -> str:
    if dur_eighths == 1:
        return ""
    if dur_eighths == 0.5:
        return "/2"
    return str(int(dur_eighths))


def carve_rest(notes, a: int, b: int):
    """Remove the tick range [a,b), splitting notes that cross its edges."""
    out = []
    for n in notes:
        s, e = n.tick, n.tick + n.dur
        if e <= a or s >= b:
            out.append(n)
            continue
        if s < a:
            out.append(Note(s, a - s, n.midi, n.vel, n.voice, n.art))
        if e > b:
            out.append(Note(b, e - b, n.midi, n.vel, n.voice, n.art))
    return out


def build_events(data, co2, contrast, start_year: int, end_year: int):
    rows = [(y, m) for y, m in data if start_year <= y <= end_year]
    if not rows:
        raise SystemExit("no temperature data in requested range")

    mel: list[Note] = []
    bass: list[Note] = []
    co2_notes: list[Note] = []
    motifs: list[Note] = []
    contrast_notes: list[Note] = []
    tempo_map: list[tuple[int, int]] = []
    dec_vols = []

    co2_start_bar = next(
        (i for i, (y, _) in enumerate(rows) if y >= 1958 and y in co2), None
    )

    prev_dec = rows[0][1][-1]

    for bar, (year, months) in enumerate(rows):
        # ---------- temperature melody: 12 slots, one per month ----------
        slot = 0
        i = 0
        while i < 12:
            t = months[i]
            prev = months[i - 1] if i > 0 else prev_dec
            delta = abs(t - prev)

            if delta > 0.22 and i < 11:
                d1 = degree_for(t)
                d2 = degree_for(t + (0.08 if t >= prev else -0.08))
                dur = int(EIGHTH // 2 * 0.55)  # staccato 16ths
                tick = bar * BAR_TICKS + slot * EIGHTH
                vel = int(clamp(60 + (t - T_MIN) / (T_MAX - T_MIN) * 55, 45, 118))
                mel.append(Note(tick, dur, midi_for(d1), vel, 0, -1))
                mel.append(Note(tick + EIGHTH // 2, dur, midi_for(d2), max(40, vel - 8), 0, -1))
                slot += 1
                i += 1
                continue

            hold = 1
            if delta < 0.06:
                hold = 2
                if (
                    i < 10
                    and abs(months[i + 1] - t) < 0.05
                    and abs(months[i + 2] - months[i + 1]) < 0.05
                ):
                    hold = 3
            hold = min(hold, 12 - slot)
            dur = hold * EIGHTH
            tick = bar * BAR_TICKS + slot * EIGHTH
            vel = int(clamp(60 + (t - T_MIN) / (T_MAX - T_MIN) * 55, 45, 118))
            art = -1 if delta > 0.15 else 0  # volatile months: crisp, marked attack
            if art == -1:
                dur = int(dur * 0.55)
            mel.append(Note(tick, dur, midi_for(degree_for(t)), vel, 0, art))
            slot += hold
            i += hold

        # ---------- syncopated bass ----------
        decade_idx = (year - start_year) // 10
        root = [45, 48, 50, 52, 55][decade_idx % 5]
        fifth, octave = root + 7, root + 12
        year_mean = sum(months) / 12
        vel = int(clamp(52 + year_mean * 22, 48, 86))

        if bar % 10 == 9:
            pattern = [(0, 12, root)]
        elif decade_idx % 2 == 0:
            # onsets 0, 3, 6, 8 — the 2+4 ending pushes into the next bar
            pattern = [
                (0, 3, root),
                (3, 3, fifth),
                (6, 2, root),
                (8, 4, octave if year_mean > 0.6 else fifth),
            ]
        else:
            # onsets 0, 4, 6, 9 — a one-eighth hole before the fifth
            pattern = [
                (0, 3, root),
                (4, 2, fifth),
                (6, 3, root),
                (9, 3, octave if year_mean > 0.6 else fifth),
            ]
        for s, d, pitch in pattern:
            bass.append(Note(bar * BAR_TICKS + s * EIGHTH, d * EIGHTH, pitch, vel, 1))

        # ---------- CO2 counter-line, one long note every two years ----------
        if (
            co2_start_bar is not None
            and bar >= co2_start_bar
            and (bar - co2_start_bar) % 2 == 0
            and bar % 10 != 9
            and year in co2
        ):
            idx = ((bar - co2_start_bar) // 2) % 5
            dur_slots = [12, 9, 12, 6, 12][idx]
            pitch = co2_midi(co2_degree(co2[year]))
            co2_notes.append(Note(bar * BAR_TICKS, dur_slots * EIGHTH, pitch, 66, 2))

        # ---------- hemispheric contrast, one high sustained note per year ----------
        if bar % 10 != 9 and year in contrast:
            cv = contrast[year]
            vel = int(clamp(50 + (cv - CONTRAST_MIN) / (CONTRAST_MAX - CONTRAST_MIN) * 35, 48, 88))
            contrast_notes.append(
                Note(
                    bar * BAR_TICKS + 6 * EIGHTH,
                    6 * EIGHTH,
                    contrast_midi(contrast_degree(cv)),
                    vel,
                    4,
                )
            )

        # ---------- tempo ----------
        if bar % 10 == 0:
            dec = rows[bar : bar + 10]
            ds = []
            for _, mm in dec:
                for k in range(1, 12):
                    ds.append(abs(mm[k] - mm[k - 1]))
            vol = sum(ds) / len(ds)
            dec_vols.append(vol)
            bpm = int(clamp(round(68 + (vol - 0.075) / 0.035 * 44), 68, 112))
            tempo_map.append((bar, bpm))

        prev_dec = months[-1]

    # ---------- form: breath rests, cadences, decade motifs ----------
    for d in range(0, len(rows), 10):
        for b in (d + 3, d + 7):
            if b < len(rows):
                a = b * BAR_TICKS + 6 * EIGHTH
                z = b * BAR_TICKS + 9 * EIGHTH
                mel = carve_rest(mel, a, z)

        cb = d + 9
        if cb >= len(rows):
            continue
        dec = rows[d : d + 10]
        dec_mean = sum(sum(m) / 12 for _, m in dec) / len(dec)

        # replace the cadence bar with a long note + a 3-note bell motif
        mel = [n for n in mel if not (cb * BAR_TICKS <= n.tick < (cb + 1) * BAR_TICKS)]
        mel.append(
            Note(
                cb * BAR_TICKS,
                6 * EIGHTH,
                midi_for(degree_for(dec_mean)),
                int(clamp(60 + (dec_mean - T_MIN) / (T_MAX - T_MIN) * 55, 50, 118)),
                0,
            )
        )
        for k in range(3):
            pitch = midi_for(degree_for(dec[0][1][k])) + 12
            motifs.append(
                Note(cb * BAR_TICKS + (6 + k) * EIGHTH, EIGHTH, pitch, 88, 3)
            )

    return rows, mel, bass, co2_notes, motifs, contrast_notes, tempo_map, dec_vols, co2_start_bar


def bar_times(tempo_map, n_bars: int):
    times = []
    t = 0.0
    idx = 0
    bpm = tempo_map[0][1]
    for bar in range(n_bars):
        while idx + 1 < len(tempo_map) and tempo_map[idx + 1][0] <= bar:
            idx += 1
            bpm = tempo_map[idx][1]
        times.append(t)
        t += 240.0 / bpm  # 12 eighths per bar
    return times, t


def write_abc(path: Path, rows, mel, bass, co2_notes, motifs, contrast_notes, tempo_map, start_year):
    def by_bar(notes):
        bars = {}
        for n in notes:
            bars.setdefault(n.tick // BAR_TICKS, []).append(n)
        for b in bars:
            bars[b].sort(key=lambda n: n.tick)
        return bars

    mel_bars = by_bar(mel)
    bass_bars = by_bar(bass)
    co2_bars = by_bar(co2_notes)
    motif_bars = by_bar(motifs)
    contrast_bars = by_bar(contrast_notes)

    def render_bar(notes, bar):
        out, slot = [], 0.0
        for n in notes:
            s = (n.tick - bar * BAR_TICKS) / EIGHTH
            if s > slot:
                out.append("z" + abc_len(s - slot))
            dur = n.dur / EIGHTH
            out.append(abc_pitch(n.midi) + abc_len(dur))
            slot = max(slot, s + dur)
        if slot < 12:
            out.append("z" + abc_len(12 - slot))
            slot = 12
        assert abs(slot - 12) < 1e-6, (bar, slot, out)
        return " ".join(out)

    lines = [
        "X:1",
        f"T:One Hundred Years of Warming, with CO2 ({start_year}-{start_year + len(rows) - 1})",
        "C:Data: NASA GISTEMP v4 + NOAA Mauna Loa CO2. Public domain.",
        "M:12/8",
        "L:1/8",
        f"Q:3/8={tempo_map[0][1]}",
        "K:Am",
        'V:1 name="Temperature (flute)" clef=treble',
        'V:2 name="Bass (cello)" clef=bass',
        'V:3 name="CO2 (bassoon)" clef=alto',
        'V:4 name="Decade motif (glockenspiel)" clef=treble',
        'V:5 name="Hemispheres (pad)" clef=treble',
        "",
        "% temperature: A minor pentatonic A3-A5; pitch=anomaly, rhythm=volatility",
        "% CO2: D Dorian D3-D4, one long note every 2 years, enters 1958",
        "% contrast: NH-SH -> A minor pentatonic A5-A6, one sustained note per year",
        "% form: 10-bar decades; breath rests at bars 4 and 8; cadence + motif at bar 10",
    ]

    dyn = ["!p!", "!mp!", "!mf!", "!f!", "!ff!"]
    for d in range(0, len(rows), 10):
        y0 = rows[d][0]
        bpm = dict(tempo_map).get(d, tempo_map[d // 10][1])
        dec = rows[d : d + 10]
        mean_t = sum(sum(m) / 12 for _, m in dec) / len(dec)
        dyn_mark = dyn[clamp(int((mean_t + 0.4) / 0.45), 0, 4)]

        lines.append(f"% ===== {y0}s =====")
        v1 = [f"[V:1] [Q:3/8={bpm}] {dyn_mark}"]
        v2, v3, v4, v5 = ["[V:2]"], ["[V:3]"], ["[V:4]"], ["[V:5]"]
        for b in range(d, min(d + 10, len(rows))):
            v1.append(render_bar(mel_bars.get(b, []), b) + " |")
            v2.append(render_bar(bass_bars.get(b, []), b) + " |")
            v3.append(render_bar(co2_bars.get(b, []), b) + " |")
            v4.append(render_bar(motif_bars.get(b, []), b) + " |")
            v5.append(render_bar(contrast_bars.get(b, []), b) + " |")
        lines.extend([" ".join(v1), " ".join(v2), " ".join(v3), " ".join(v4), " ".join(v5), ""])

    path.write_text("\n".join(lines) + "\n")


def write_midi(path: Path, notes, tempo_map):
    mid = mido.MidiFile(type=1, ticks_per_beat=TPQ)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="tempo/meta", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=12, denominator=8, time=0))
    meta.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(1.5 * tempo_map[0][1]), time=0)
    )
    last_bar = 0
    for bar, bpm in tempo_map[1:]:
        meta.append(
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(1.5 * bpm),
                time=(bar - last_bar) * BAR_TICKS,
            )
        )
        last_bar = bar
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    programs = [
        (0, 73, "temperature/flute"),
        (1, 42, "bass/cello"),
        (2, 70, "co2/bassoon"),
        (3, 9, "motif/glockenspiel"),
        (4, 89, "contrast/warm pad"),
    ]
    for voice, program, name in programs:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        tr.append(mido.Message("program_change", channel=voice, program=program, time=0))
        events = []
        for n in notes:
            if n.voice != voice:
                continue
            events.append(
                (n.tick, 0, mido.Message("note_off", channel=voice, note=n.midi, velocity=0))
            )
            events.append(
                (n.tick, 1, mido.Message("note_on", channel=voice, note=n.midi, velocity=n.vel))
            )
            events.append(
                (
                    n.tick + n.dur,
                    0,
                    mido.Message("note_off", channel=voice, note=n.midi, velocity=0),
                )
            )
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _, msg in events:
            tr.append(msg.copy(time=tick - last))
            last = tick
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)

    mid.save(path)


def write_song_json(path: Path, rows, notes, tempo_map, co2, contrast, start_year, co2_start_bar):
    times, total = bar_times(tempo_map, len(rows))
    bars = []
    for i, (year, _) in enumerate(rows):
        d = times[i + 1] - times[i] if i + 1 < len(times) else total - times[i]
        bars.append({"year": year, "t": round(times[i], 4), "d": round(d, 4)})

    out_notes = []
    for n in notes:
        bar = n.tick // BAR_TICKS
        bar_dur = bars[bar]["d"]
        t = bars[bar]["t"] + (n.tick - bar * BAR_TICKS) / BAR_TICKS * bar_dur
        o = {
            "t": round(t, 4),
            "d": round(n.dur / BAR_TICKS * bar_dur, 4),
            "m": n.midi,
            "v": n.vel,
            "vc": n.voice,
        }
        if n.art:
            o["art"] = n.art
        out_notes.append(o)
    out_notes.sort(key=lambda n: (n["t"], n["vc"], n["m"]))

    sections = []
    for d in range(0, len(rows), 10):
        dec = rows[d : d + 10]
        mean_t = sum(sum(m) / 12 for _, m in dec) / len(dec)
        sections.append(
            {
                "year": dec[0][0],
                "start": d,
                "end": min(d + 10, len(rows)),
                "mean": round(mean_t, 3),
            }
        )

    song = {
        "meta": {
            "title": "One Hundred Years of Warming",
            "subtitle": "NASA GISTEMP × NOAA Mauna Loa CO₂",
            "start": start_year,
            "end": rows[-1][0],
            "duration": round(total, 3),
            "co2Entry": rows[co2_start_bar][0] if co2_start_bar is not None else None,
            "counts": {
                VOICE_NAMES[v]: sum(1 for n in notes if n.voice == v) for v in range(5)
            },
        },
        "bars": bars,
        "tempo": [[b, v] for b, v in tempo_map],
        "sections": sections,
        "series": {
            "years": [y for y, _ in rows],
            "temp": [round(sum(m) / 12, 3) for _, m in rows],
            "co2": [round(co2[y], 2) if y in co2 else None for y, _ in rows],
            "contrast": [round(contrast[y], 3) if y in contrast else None for y, _ in rows],
        },
        "notes": out_notes,
    }
    path.write_text(
        "window.SONG = " + json.dumps(song, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    return song


def write_chart(path: Path, rows, notes, co2, contrast, start_year, co2_start_bar):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [y for y, _ in rows]
    annual = [sum(m) / 12 for _, m in rows]
    co2_years = [y for y, _ in rows if y in co2]
    co2_vals = [co2[y] for y in co2_years]
    contrast_years = [y for y in years if y in contrast]
    contrast_vals = [contrast[y] for y in contrast_years]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.4), gridspec_kw={"height_ratios": [1, 1.55]}, sharex=True
    )
    ax1.plot(years, annual, color="#e74c3c", lw=2.2, label="Temperature anomaly")
    ax1.axhline(0, color="#7f8c8d", lw=0.8, ls="--")
    ax1.fill_between(years, annual, 0, color="#e74c3c", alpha=0.10)
    ax1.set_ylabel("°C vs 1951-80", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_title("Temperature, CO₂ and NH−SH contrast → music")

    if contrast_vals:
        ax1.plot(
            contrast_years, contrast_vals, color="#9b59b6", lw=1.4, ls="--",
            label="NH−SH contrast",
        )

    ax1b = ax1.twinx()
    ax1b.plot(co2_years, co2_vals, color="#16a085", lw=2.0, label="CO₂")
    ax1b.set_ylabel("CO₂ (ppm)", color="#16a085")
    ax1b.tick_params(axis="y", labelcolor="#16a085")
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper left", fontsize=8, frameon=False)

    colors = {0: "#e67e22", 1: "#3498db", 2: "#1abc9c", 3: "#f1c40f", 4: "#9b59b6"}
    alpha = {0: 0.85, 1: 0.55, 2: 0.9, 3: 0.9, 4: 0.85}
    for n in notes:
        x = start_year + n.tick / BAR_TICKS
        w = n.dur / BAR_TICKS
        ax2.add_patch(
            plt.Rectangle(
                (x, n.midi - 0.35), w, 0.7, color=colors[n.voice], alpha=alpha[n.voice], lw=0
            )
        )
    ax2.set_ylabel("MIDI pitch")
    ax2.set_xlabel("Year")
    ax2.set_xlim(start_year, start_year + len(rows))
    ax2.grid(axis="y", color="#ecf0f1", lw=0.6)
    ax2.axhline(CO2_ROOT, color="#1abc9c", lw=0.4, ls=":")
    ax2.axhline(CO2_ROOT + 12, color="#1abc9c", lw=0.4, ls=":")
    if co2_start_bar is not None:
        ax2.axvline(start_year + co2_start_bar, color="#16a085", lw=1.0, ls="--", alpha=0.7)
        ax2.text(
            start_year + co2_start_bar + 1,
            CO2_ROOT - 5,
            "CO₂ enters",
            color="#16a085",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def render_audio(mid_path: Path, wav_path: Path):
    if not Path(SF2).exists():
        print("soundfont missing, skipping audio")
        return False
    cmd = [
        "fluidsynth",
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        "22050",
        "-g",
        "0.9",
        "-o",
        "synth.reverb.active=1",
        "-o",
        "synth.reverb.room-size=0.6",
        "-o",
        "synth.reverb.level=0.5",
        SF2,
        str(mid_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1925)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--out", default="warming")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    data = load_gistemp(HERE / "gistemp.csv")
    co2 = load_co2(HERE / "co2_mm_mlo.csv")
    contrast = load_contrast(
        HERE / "gistemp_nh.csv", HERE / "gistemp_sh.csv", args.start, args.end
    )
    rows, mel, bass, co2_notes, motifs, contrast_notes, tempo_map, vols, co2_start_bar = build_events(
        data, co2, contrast, args.start, args.end
    )
    all_notes = mel + bass + co2_notes + motifs + contrast_notes

    abc_path = HERE / f"{args.out}.abc"
    mid_path = HERE / f"{args.out}.mid"
    png_path = HERE / f"{args.out}.png"
    wav_path = HERE / f"{args.out}.wav"
    json_path = HERE / "player_data.js"

    write_abc(abc_path, rows, mel, bass, co2_notes, motifs, contrast_notes, tempo_map, args.start)
    write_midi(mid_path, all_notes, tempo_map)
    write_chart(png_path, rows, all_notes, co2, contrast, args.start, co2_start_bar)
    song = write_song_json(
        json_path, rows, all_notes, tempo_map, co2, contrast, args.start, co2_start_bar
    )

    print(f"years      : {rows[0][0]}-{rows[-1][0]} ({len(rows)})")
    print(
        "notes      : "
        + ", ".join(f"{VOICE_NAMES[v]}={sum(1 for n in all_notes if n.voice == v)}" for v in range(5))
    )
    print(f"co2 entry  : {song['meta']['co2Entry']}")
    print(f"duration   : {song['meta']['duration']:.1f}s")
    print(f"tempo map  : {tempo_map}")
    print(f"decade vol : {[round(v, 3) for v in vols]}")
    print(f"wrote      : {abc_path.name}, {mid_path.name}, {png_path.name}, {json_path.name}")
    if not args.no_audio and render_audio(mid_path, wav_path):
        print(f"audio      : {wav_path.name} ({wav_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
