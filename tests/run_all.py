"""Run all.    python -m plantforge.tests.run_all   (from /home/coder)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from plantforge.tests import test_plantforge


def main():
    print("PLANTFORGE — procedural control-plant corpus for in-context SysID:")
    test_plantforge._run_all()


if __name__ == "__main__":
    main()
