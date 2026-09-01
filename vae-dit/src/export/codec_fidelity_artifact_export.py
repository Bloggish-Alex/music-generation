"""Copy codec-fidelity raw status bundles into one public run directory."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil
@dataclass(frozen=True)
class CodecFidelityArtifactExportConfig: source_dir: Path; output_dir: Path
def export_codec_fidelity_artifacts(config: CodecFidelityArtifactExportConfig)->None:
    config.output_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","validation","excluded_unpaired"):
        status=config.source_dir/f"codec_fidelity__raw_status__{split}.v1.json"
        if not status.is_file(): continue
        shutil.copyfile(status,config.output_dir/status.name)
        import json
        value=json.loads(status.read_text(encoding="utf-8"))
        for ref in value.get("artifacts",{}).values():
            source=config.source_dir/ref["path"]
            if source.is_file(): shutil.copyfile(source,config.output_dir/source.name)
        if value.get("status")=="AVAILABLE":
            obs=json.loads((config.source_dir/value["artifacts"]["observation"]["path"]).read_text(encoding="utf-8")); ref=obs.get("source_raw",{})
            if ref.get("path") and (config.source_dir/ref["path"]).is_file(): shutil.copyfile(config.source_dir/ref["path"],config.output_dir/ref["path"])
