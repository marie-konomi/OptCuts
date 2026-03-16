import os
import sys
import argparse

# スクリプト基準でプロジェクトルートと build を絶対パスに
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build = os.path.abspath(os.path.join(_root, "build"))

_so_files = [f for f in (os.listdir(_build) if os.path.isdir(_build) else []) if f.startswith("optcuts.") and f.endswith(".so")]

if os.path.isdir(_build) and _build not in sys.path:
    sys.path.insert(0, _build)

output_dir = os.path.abspath("output")

import optcuts

optcuts.set_output_folder(output_dir + os.sep)
mesh_path = os.path.join(_root, "input", "bimba_i_f10000.obj")
argv_list = [
        "optcuts",
        "100",
        mesh_path,
        str(0.99),
        str(1.0),
        str(0),
        str(4.1),
        "1",
        str(1.1),
        str(1.1),
        str(0),
    ]

ret = optcuts.run_main(argv_list)

UV, F = optcuts.get_result_mesh()
print(f"反復: {optcuts.get_iter_num()}, 結果: UV {UV.shape}, F {F.shape}")

if ret != 0:
    print(f"run_main が 0 以外で終了: {ret}")
    #sys.exit(ret)
    UV, F = optcuts.get_result_mesh()
    print(f"反復: {optcuts.get_iter_num()}, 結果: UV {UV.shape}, F {F.shape}")