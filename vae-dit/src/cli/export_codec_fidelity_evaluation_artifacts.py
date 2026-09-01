from __future__ import annotations
import argparse
from pathlib import Path
from export.codec_fidelity_artifact_export import CodecFidelityArtifactExportConfig,export_codec_fidelity_artifacts
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();export_codec_fidelity_artifacts(CodecFidelityArtifactExportConfig(a.source_dir,a.output_dir))
if __name__=='__main__':main()
