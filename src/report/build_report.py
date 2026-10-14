"""Deterministic situation facts from cached inference receipts only."""
from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from src.data.incident import DISCLAIMER, load_incident

BANNER = "SYNTHETIC TEST INPUT, NOT A MODEL RESULT"
MISSING = "not available"
NUMBER = re.compile(r"\d+(?:\.\d+)?")
LIMITATIONS = (
    "The water model has a processing-domain gap on terrain-corrected, Lee-filtered imagery. "
    "Probabilities must not be treated as calibrated confidence without calibration diagnostics. "
    "OSM edits can be stale and coverage incomplete. Road access scenarios ignore directional and "
    "vehicle restrictions and use heuristic travel times. Inundation is not structural failure. "
    "Satellite revisit timing can miss short-lived flooding. Ranking reflects mapped and satellite "
    "evidence only; it is not a measure of humanitarian need."
)
COMPARISON_NOTE = ("Post-hoc comparison, not used by the analysis. Debris-candidate and union rows were added "
                   "after the flood-only result; all rows are shown.")
COMPARISON_HEADERS = ("Variant", "IoU", "Precision", "Recall", "F1", "Our km²", "Reference km²",
                      "Overlap km²", "Tolerance precision", "Tolerance recall")


def comparison_facts(value: dict[str, Any] | None) -> dict[str, Any]:
    """Format supplied comparison facts without selecting, scoring or changing rows."""
    if value is None:
        return dict(paragraphs=["No validation comparison supplied."], headers=[], rows=[])
    variants = value.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("Comparison variants must be a JSON object")
    order = {"flood_mask": 0, "debris_candidate": 1, "union": 2}
    rows = []
    for name in sorted(variants, key=lambda key: (order.get(key, 3), key)):
        row = variants[name]
        rows.append([clean_text(name), *[format_number(row.get("metrics", {}).get(key)) for key in ("iou", "precision", "recall", "f1")],
                     *[format_number(row.get("areas_km2", {}).get(key)) for key in ("ours", "reference", "overlap")],
                     *[format_number(row.get("tolerance_metrics", {}).get(key)) for key in ("precision_tol", "recall_tol")]])
    domain = value.get("domain", {})
    paragraphs = [COMPARISON_NOTE,
                  f"Analysed domain area: {format_number(domain.get('area_km2'))} km². Share of our flood outside the analysed domain: {format_number(domain.get('our_flood_outside_share'), percent=True)}."]
    if value.get("params", {}).get("tolerance_m") is not None:
        paragraphs.append(f"Declared tolerance: {format_number(value['params']['tolerance_m'])} m.")
    types = value.get("reference_event_types", [])
    if isinstance(types, dict):
        types = [item for values in types.values() for item in (values if isinstance(values, list) else [values])]
    if types:
        paragraphs.append("These reference polygons are labelled " + ", ".join(sorted({clean_text(item) for item in types}))
                          + "; this is a proxy check, not a flood-extent validation.")
    paragraphs.extend("Comparison warning: " + clean_text(warning) for warning in value.get("warnings", []))
    return dict(paragraphs=paragraphs, headers=list(COMPARISON_HEADERS), rows=rows)


def clean_text(value: Any) -> str:
    """Retain Unicode; replace prohibited outcome language in supplied prose."""
    text = MISSING if value is None else str(value)
    for pattern, replacement in ((r"destroyed", "potentially affected"),
                                 (r"casualt\w*|deaths?", "unverified outcomes"),
                                 (r"population", "demographic")):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def format_number(value: Any, *, percent: bool = False) -> str:
    """Finite numbers: up to four decimals, no grouping; fractions x100 for percent.

    Dates, ids and supplied sentences retain their original numeric tokens.
    Missing/nonfinite values are never replaced by a numeric estimate.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return MISSING
    number = value * 100 if percent else value
    formatted = f"{number:.4f}".rstrip("0").rstrip(".")
    return (formatted or "0") + ("%" if percent else "")


def numbers_trace(text: str, sources: Iterable[Any]) -> dict[str, list[str]]:
    """Audit visible prose against numeric values/formats and tokens in source strings.

    HTML callers pass visible text, excluding scripts, styles and attributes.
    Collection cardinalities and the count of baseline-only uncertain settlement
    features are permitted deterministic counts. This checks token provenance,
    not the semantic correctness of a claim.
    """
    allowed: set[str] = set()
    gap_ids: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for gap in value.get("data_gaps", []) if isinstance(value.get("data_gaps"), list) else []:
                if isinstance(gap, dict) and gap.get("id"):
                    gap_ids.add(gap["id"])
            if value.get("type") == "FeatureCollection":
                gaps = [feature for feature in value.get("features", []) if _data_gap(feature.get("properties", {}))]
                gap_ids.update(feature["properties"]["osm_id"] for feature in gaps if feature["properties"].get("osm_id"))
                allowed.update(NUMBER.findall(format_number(len(gaps))))
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple)):
            allowed.update(NUMBER.findall(format_number(len(value))))
            for item in value:
                visit(item)
        elif isinstance(value, str):
            allowed.update(NUMBER.findall(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            allowed.update(NUMBER.findall(str(value)))
            allowed.update(NUMBER.findall(format_number(value)))
            allowed.update(NUMBER.findall(format_number(value, percent=True)))

    for source in sources:
        visit(source)
    allowed.update(NUMBER.findall(format_number(len(gap_ids))))
    tokens = NUMBER.findall(text)
    return {"tokens": tokens, "untraced": sorted(set(tokens) - allowed)}


def read_json(path: Path) -> Any:
    """Read a required UTF-8 artifact with a useful error."""
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def latest_directory(root: Path, filename: str) -> Path:
    """Choose the newest complete receipt; path name breaks timestamp ties."""
    candidates = [path.parent for path in root.glob(f"*/{filename}") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No cached {filename} under {root}; run the analysis first")
    return max(candidates, key=lambda path: ((path / filename).stat().st_mtime_ns, path.name))


def _layer_path(value: str, incident_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else incident_dir / path


def _eligible(row: dict[str, Any]) -> bool:
    """Display guard for legacy priorities; never recompute or change source scores."""
    if row.get("status") == "CUT OFF":
        return True
    baseline = row.get("baseline_connected", {})
    evidence = [item for item in row.get("evidence", []) if item.get("tier", 0) > 0
                and item.get("link_id") in row.get("responsible_links", [])]
    if not row.get("snapped") or not evidence:
        return False
    if row.get("access_detail") == "degraded" or "degraded" in row.get("access_by_class", {}).values():
        return True
    return row.get("status") == "UNCERTAIN" and (
        any(baseline.get(kind) for kind in row.get("possible_cut_from", []))
        or (all(baseline.get(kind) for kind in ("hospital", "town"))
            and any(item["tier"] == 1 for item in evidence)))


def _data_gap(row: dict[str, Any]) -> bool:
    """Baseline-only uncertainty, with no qualifying flood-attributable priority."""
    return (row.get("status") == "UNCERTAIN" and not _eligible(row) and
            (not row.get("snapped") or not all(row.get("baseline_connected", {}).get(kind) for kind in ("hospital", "town"))))


def build_report_model(incident: str | Path, impact_dir: str | Path | None = None,
                       flood_dir: str | Path | None = None, *,
                       cache_dir: str | Path = "data/cache", validation_json: str | Path | None = None) -> dict[str, Any]:
    """Read receipts, retain source facts and format one shared report model.

    Synthetic impact never borrows real flood statistics. Missing rasters are
    permitted for reporting, but missing computed JSON/GeoJSON is an error.
    """
    receipt = load_incident(incident, cache_dir=cache_dir)
    validation = read_json(Path(validation_json)) if validation_json is not None else None
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError("Comparison input must be a JSON object")
        if validation.get("incident_id") and validation["incident_id"] != receipt.incident_id:
            raise ValueError("Comparison input belongs to a different incident")
    directory = Path(receipt.cache_path).parent
    impact_path = Path(impact_dir) if impact_dir else latest_directory(directory / "impact", "impact_meta.json")
    if not impact_path.is_dir():
        raise FileNotFoundError(f"Impact directory is missing: {impact_path}")
    if not (impact_path / "impact_meta.json").is_file():
        impact_path = latest_directory(impact_path, "impact_meta.json")
    impact = read_json(impact_path / "impact_meta.json")
    if impact.get("incident_id") != receipt.incident_id:
        raise ValueError("Impact receipt belongs to a different incident")
    priorities = read_json(impact_path / "priorities.json")
    chokes = read_json(impact_path / "chokepoints.json")
    clusters = read_json(impact_path / "clusters.json")
    geo = {name: read_json(impact_path / f"{name}.geojson")
           for name in ("damaged_assets", "settlements", "chokepoints")}
    synthetic = (impact.get("banner") == BANNER or impact.get("flood_params", {}).get("synthetic") is True
                 or any(flag.get("code") == "synthetic_flood_input" for flag in impact.get("flags", [])))
    notices: list[str] = []
    flood: dict[str, Any] = {}
    flood_path: Path | None = None
    if synthetic:
        if flood_dir is not None:
            raise ValueError("Synthetic impact cannot be combined with a real flood directory; omit --flood-dir")
        notices.append("Synthetic test-box boundary shown; a saved synthetic flood raster and flood statistics are not available.")
    else:
        linked = impact.get("flood_path")
        flood_path = (Path(flood_dir) if flood_dir else _layer_path(linked, directory).parent if linked
                      else latest_directory(directory / "flood", "flood_meta.json"))
        if flood_path.is_file():
            flood_path = flood_path.parent
        if not (flood_path / "flood_meta.json").is_file():
            flood_path = latest_directory(flood_path, "flood_meta.json")
        flood = read_json(flood_path / "flood_meta.json")
        flood_incident = flood.get("params", {}).get("incident_id")
        if flood_incident and flood_incident != receipt.incident_id:
            raise ValueError("Flood receipt belongs to a different incident")
        used_key = impact.get("flood_params", {}).get("cache_key")
        if used_key and flood.get("params", {}).get("cache_key") != used_key:
            raise ValueError("Flood directory does not match the flood used for this impact analysis")
        for name in ("flood_extent", "debris_candidates"):
            geo[name] = read_json(flood_path / f"{name}.geojson")
    flags: list[dict[str, Any]] = []
    seen = set()
    for source in (receipt.quality, flood, impact):
        for flag in source.get("flags", []):
            key = (flag.get("code"), flag.get("message"), flag.get("severity"))
            if key not in seen:
                flags.append(flag)
                seen.add(key)
    flags.sort(key=lambda row: ({"limiting": 0, "caution": 1, "info": 2}.get(row.get("severity"), 3),
                                row.get("code", ""), row.get("message", "")))
    no_flood = not synthetic and (flood.get("stats", {}).get("flood_area_km2") == 0
                                  or any(flag.get("code") == "no_flood_detected" for flag in flags))
    subjects = {feature["properties"]["osm_id"]: feature["properties"] for feature in geo["settlements"]["features"]}
    ranked = [row for row in priorities.get("settlements", []) if not no_flood and _eligible(subjects.get(row["id"], {}))]
    ranked.sort(key=lambda row: (-row["score"], row["id"]))
    gaps = {row["id"]: row for row in priorities.get("data_gaps", [])}
    for identifier, row in subjects.items():
        if _data_gap(row):
            gaps.setdefault(identifier, {"id": identifier, "name": row.get("name"), "reason": row.get("reason"),
                                         "status": row["status"]})
    if len(ranked) != len(priorities.get("settlements", [])):
        notices.append("Legacy or non-attributable priorities hidden; original scores are preserved and data gaps shown separately.")
    damaged = sorted((row for row in chokes.get("damaged", []) if row.get("count", 0) > 0),
                     key=lambda row: (-row["count"], row["link_id"]))
    latent = sorted((row for row in chokes.get("latent", []) if row.get("count", 0) > 0),
                    key=lambda row: (-row["count"], row["link_id"]))
    zero_links = {row["link_id"]: row for row in [*chokes.get("damaged_no_leverage", []),
                                                *chokes.get("damaged", [])] if row.get("count", 0) <= 0}
    valid_links = {row["link_id"] for row in damaged}
    ranked_links = sorted((row for row in priorities.get("links", []) if not no_flood and row["id"] in valid_links),
                          key=lambda row: (-row["score"], row["id"]))
    osm_meta: dict[str, Any] = {}
    if receipt.layers.get("osm_metadata"):
        candidate = _layer_path(receipt.layers["osm_metadata"], directory)
        if candidate.is_file():
            osm_meta = read_json(candidate)
    pair, aoi = receipt.pair_summary, receipt.aoi
    stats = flood.get("stats", {})
    counts = impact.get("counts", {}).get("counts", {})
    tallies = impact.get("status_tallies", {})
    f = format_number
    gap_note = (f"{f(len(gaps))} settlements are UNCERTAIN because mapped OSM roads show no baseline access to a hospital or town. "
                "This is a data gap, not flood damage.") if gaps else None
    priorities_note = ("Ranked from flood-evidence and OSM-derived access scores. "
                       f"CUT OFF settlements: {f(tallies.get('CUT OFF'))}. This is not an isolation ranking."
                       + (" No settlement is CUT OFF in this run." if tallies.get("CUT OFF") == 0 else ""))
    findings = [
        "No flood detected." if no_flood else f"Flood extent: {f(stats.get('flood_area_km2'))} km²; valid mapped area: {f(stats.get('valid_area_km2'))} km².",
        "Flood tiers: " + "; ".join(f"{tier} {f(stats.get('area_per_confidence_tier_km2', {}).get(tier))} km²" for tier in ("strong", "probable", "uncertain")) + ".",
        f"Pre-existing water: {f(stats.get('preexisting_water_km2'))} km²; terrain-excluded new water: {f(stats.get('terrain_excluded_km2'))} km²; debris candidates: {f(stats.get('debris_candidate_km2'))} km².",
        "Potentially affected assets: " + "; ".join(f"{kind} {f(counts.get(kind))}" for kind in ("roads", "bridges", "buildings", "hospitals")) + ". Bridges are included in roads.",
        "Mapped settlement status: " + "; ".join(f"{kind} {f(tallies.get(kind))}" for kind in ("CUT OFF", "UNCERTAIN", "CONNECTED")) + ". Baseline gaps do not establish flood isolation.",
        ("Damaged-link restoration: " + " ".join(clean_text(row.get("sentence")) for row in damaged[:2] if row.get("sentence"))) if damaged else "No damaged links with demonstrated single-link restoration leverage.",
        ("Flood-attributable priorities: " + "; ".join(clean_text(row.get("name") or row["id"]) + ": " + ", ".join(clean_text(reason) for reason in row.get("why", [])) for row in ranked[:2])) if ranked else "No flood-attributable priorities.",
        ("Latent bridge risk: " + " ".join(clean_text(row["sentence"]) for row in latent[:1] if row.get("sentence"))) if latent else "No latent critical bridges with demonstrated access loss in the saved analysis.",
    ]
    training = next((clean_text(flag["message"]) for flag in flags if flag.get("code") == "domain_gap"),
                    "Training processing-chain details are not available in this receipt.")
    provenance_lines = ["; ".join(clean_text(row.get(key)) for key in ("role", "source") if row.get(key))
                        + ": " + clean_text(row.get("acquisition_date") or row.get("snapshot_date"))
                        + ("; snapshot verified: " + str(row.get("snapshot_verified")) if row.get("role") == "osm" else "")
                        for row in receipt.provenance.get("inputs", [])]
    report = {
        "title": clean_text(aoi.get("name") or receipt.provenance.get("request", {}).get("place") or receipt.incident_id),
        "incident": receipt.incident_id, "banner": BANNER if synthetic else None,
        "incident_lines": [f"Area: {f(aoi.get('area_km2'))} km²; flood date: {clean_text(aoi.get('flood_date'))}.",
                           f"Imagery: {clean_text(pair.get('orbit_state'))} orbit {f(pair.get('relative_orbit'))}; {clean_text(pair.get('pre_date'))} → {clean_text(pair.get('post_date'))}; {f(pair.get('days_after_flood'))} days after flood.",
                           f"Incident generation (UTC): {clean_text(receipt.created_utc)}; report uses saved receipts.",
                           f"Trust: {clean_text(impact.get('trust_summary') or receipt.quality.get('trust_summary'))}."],
        "findings": findings, "data_gap_note": gap_note, "priorities_note": priorities_note,
        "comparison": comparison_facts(validation),
        "limitations": [training, LIMITATIONS, *[clean_text(flag.get("message")) for flag in flags if flag.get("severity") in ("caution", "limiting")]],
        "provenance": provenance_lines, "disclaimer": DISCLAIMER,
    }
    return dict(report=report, incident=receipt.to_dict(), impact=impact, flood=flood, geo=geo, clusters=clusters,
                priorities=dict(settlements=ranked, links=ranked_links, data_gaps=sorted(gaps.values(), key=lambda row: row["id"]),
                                message=priorities.get("message") if ranked else "No flood-attributable priorities"),
                chokepoints=dict(damaged=damaged, latent=latent, damaged_no_leverage=sorted(zero_links.values(), key=lambda row: row["link_id"])),
                flags=flags, notices=notices, osm_metadata=osm_meta, synthetic=synthetic, no_flood=no_flood,
                validation=validation,
                paths=dict(incident_dir=str(directory), impact_dir=str(impact_path), flood_dir=str(flood_path) if flood_path else None,
                           validation_json=str(Path(validation_json).resolve()) if validation_json is not None else None),
                trace_sources=[receipt.to_dict(), impact, flood, priorities, chokes, clusters, *geo.values(), osm_meta,
                               *([validation] if validation is not None else [])])


def report_sections(model: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Both text renderers use identical section contents."""
    report = model["report"]
    return [("Incident", [report["incident"], *report["incident_lines"]]),
            ("Key findings", [*report["findings"], *([report["data_gap_note"]] if report["data_gap_note"] else [])]),
            ("Priorities", [report["priorities_note"]]),
            ("Post-hoc comparison", report["comparison"]["paragraphs"]),
            ("Confidence and limitations", report["limitations"]),
            ("Provenance", report["provenance"]), ("Disclaimer", [report["disclaimer"]])]


def render_report_md(model: dict[str, Any]) -> str:
    """Plain UTF-8 Markdown; user text is inert and HTML-escaped."""
    report = model["report"]
    lines = ["# " + escape(report["title"]), ""]
    if report["banner"]:
        lines.extend(["**" + report["banner"] + "**", ""])
    for title, paragraphs in report_sections(model):
        lines.extend(["## " + title, ""])
        for paragraph in paragraphs:
            lines.extend([escape(clean_text(paragraph)), ""])
        if title == "Post-hoc comparison" and report["comparison"]["headers"]:
            table = report["comparison"]
            lines.extend(["| " + " | ".join(table["headers"]) + " |",
                          "| " + " | ".join("---" for _ in table["headers"]) + " |",
                          *["| " + " | ".join(escape(cell).replace("|", "&#124;").replace("\n", " ") for cell in row) + " |" for row in table["rows"]], ""])
    return "\n".join(lines)


def render_report_html(model: dict[str, Any]) -> str:
    """Print CSS targets a single A4 page; content never uses invented totals."""
    report = model["report"]
    banner = f'<div class="banner">{report["banner"]}</div>' if report["banner"] else ""
    sections = ""
    for title, paragraphs in report_sections(model):
        sections += f'<section><h2>{title}</h2>' + "".join(f'<p>{escape(clean_text(p))}</p>' for p in paragraphs)
        if title == "Post-hoc comparison":
            sections += comparison_table_html(report["comparison"])
        sections += '</section>'
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(report["title"])}</title><style>'
            '@page{size:A4;margin:12mm}*{box-sizing:border-box}body{font:9pt/1.3 system-ui,sans-serif;color:#142130;max-width:186mm;margin:auto}'
            'h1{font-size:19pt;margin:0 0 8px}h2{font-size:10pt;margin:8px 0 3px;text-transform:uppercase}p{margin:3px 0;overflow-wrap:anywhere}'
            '.banner{background:#a31823;color:white;padding:8px;font-weight:bold}section{break-inside:avoid}'
            'table{width:100%;border-collapse:collapse;font-size:7pt}th,td{text-align:left;border-bottom:1px solid #cbd5df;padding:2px;overflow-wrap:anywhere}'
            '@media screen{body{padding:20px;background:#fff}}@media print{body{padding:0}a{color:inherit}}'
            f'</style></head><body><h1>{escape(report["title"])}</h1>{banner}{sections}</body></html>')


def comparison_table_html(table: dict[str, Any]) -> str:
    """Render the shared comparison table with escaped, preformatted source facts."""
    if not table.get("headers"):
        return ""
    return ('<div class="table-wrap"><table><thead><tr>'
            + ''.join(f'<th scope="col">{escape(cell)}</th>' for cell in table['headers'])
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>' for row in table['rows'])
            + '</tbody></table></div>')
