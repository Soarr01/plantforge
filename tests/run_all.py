"""Run all.    python -m plantforge.tests.run_all   (from the package's parent directory)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from plantforge.tests import (
    test_plantforge, test_realbench, test_aggregate, test_baselines, test_ident_exp,
    test_ablation, test_leave_one_out,
)


def main():
    print("PLANTFORGE — procedural control-plant corpus for in-context SysID:")
    test_plantforge._run_all()
    print("PLANTFORGE realbench -- offline decimation/windowing tests:")
    test_realbench._run_all()
    print("PLANTFORGE aggregate -- offline helper tests:")
    test_aggregate._run_all()
    print("PLANTFORGE baselines -- offline tests:")
    test_baselines._run_all()
    print("PLANTFORGE ident_exp -- offline tests:")
    test_ident_exp._run_all()
    print("PLANTFORGE ablation -- offline tests:")
    test_ablation._run_all()
    print("PLANTFORGE leave_one_out -- offline tests:")
    test_leave_one_out._run_all()


if __name__ == "__main__":
    main()
