from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cli.export_codec_fidelity_evaluation_artifacts import main
if __name__=='__main__':main()
