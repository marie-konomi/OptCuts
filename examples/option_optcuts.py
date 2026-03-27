#!/usr/bin/env python3
"""
OptCuts Python 例: run_main / run_from_files の CLI。
main.cpp の argv に対応: argv[1]=progMode, [2]=mesh, [3]=lambda, [4]=testID,
[5]=methodType, [6]=upperBound, [7]=bijectiveParam, [8]=sigma1, [9]=sigma2, [10]=initCutOption.
デフォルトは batch.py と同じ: progMode=10(offline), lambda=0.99, sigma1=1.0, sigma2=1.0, upperBound=4.1, scaffolding=ON.
"""
import os
import sys
import argparse

# スクリプト基準でプロジェクトルートと build を絶対パスに
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build = os.path.abspath(os.path.join(_root, "build"))

# build 内の .so が cpython-39 用だけのときは、python3.9 で再実行する
_so_files = [f for f in (os.listdir(_build) if os.path.isdir(_build) else []) if f.startswith("optcuts.") and f.endswith(".so")]
_need_39 = any("cpython-39" in f for f in _so_files) and sys.version_info[:2] != (3, 9)
if _need_39:
    import shutil
    for _name in ("python3.9", "python3.9.exe"):
        _exe = shutil.which(_name)
        if not _exe and _name == "python3.9":
            _pyenv = os.path.expanduser("~/.pyenv/versions/3.9.9/bin/python")
            if os.path.isfile(_pyenv):
                _exe = _pyenv
        if _exe:
            os.execv(_exe, [_exe, os.path.abspath(__file__)] + sys.argv[1:])
    print("optcuts 用の .so は Python 3.9 用です。python3.9 を入れるか pyenv で 3.9 を有効にしてください。")
    sys.exit(1)

# optcuts は build 内の .so から読む
if os.path.isdir(_build) and _build not in sys.path:
    sys.path.insert(0, _build)

try:
    import optcuts
except ImportError as e:
    if not _so_files:
        print("optcuts を import できません。build に optcuts.*.so がありません。")
        print("  CMake で -DOPTCUTS_BUILD_PYTHON=ON にしてビルドしてください。")
    else:
        print("optcuts を import できません。")
        print(f"  詳細: {e}")
        print(f"  build 内の .so: {', '.join(_so_files)}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="OptCuts: run_main (main.cpp 相当) または run_from_files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        default=os.path.join(_root, "input", "benchmark", "cat.obj"),
        help="入力メッシュ .obj (default: input/benchmark/cat.obj)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(_root, "output"),
        help="出力ディレクトリ (default: output)",
    )
    # batch.py 相当: lambda=0.99, sigma1=1.0, sigma2=1.0, upperBound=4.1, scaffolding=ON, progMode=10(offline)
    parser.add_argument(
        "--lambda", "-l",
        type=float,
        default=0.99,
        dest="lambda_init",
        help="lambda_init 0~1 (default: 0.99, batch.py 相当)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        dest="offline_mode",
        help="progMode=10 で offline 実行 (default: True, batch.py 相当)",
    )
    parser.add_argument(
        "--no-offline",
        action="store_false",
        dest="offline_mode",
        help="progMode=0 で通常 GUI",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="progMode=100 で headless 実行",
    )
    parser.add_argument(
        "--test-id",
        type=float,
        default=1.0,
        help="testID (default: 1.0)",
    )
    parser.add_argument(
        "--method",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="methodType: 0=OptCuts, 1=OptCuts_noDual, 2=EBCuts, 3=DistMin (default: 0)",
    )
    parser.add_argument(
        "--upper-bound", "-b",
        type=float,
        default=4.1,
        help="upperBound (default: 4.1, batch.py 相当)",
    )
    parser.add_argument(
        "--no-scaffolding",
        action="store_true",
        help="bijectiveParam=0 (scaffolding オフ)",
    )
    parser.add_argument(
        "--sigma1",
        type=float,
        default=1.0,
        help="sigma1 上限 (default: 1.0, batch.py 相当)",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=1.0,
        help="sigma2 上限 (default: 1.0, batch.py 相当)",
    )
    parser.add_argument(
        "--init-cut",
        type=int,
        default=0,
        choices=[0, 1],
        help="initCutOption: 0=random 2-edge, 1=farthest 2-point (default: 0, batch.py 相当)",
    )
    parser.add_argument(
        "--from-files",
        action="store_true",
        help="run_from_files を使う（run_main の代わり）",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="run_from_files 用: max_iter (default: 100)",
    )
    parser.add_argument(
        "--max-topo-iters",
        type=int,
        default=20,
        help="run_from_files 用: max_topo_iters (default: 20)",
    )
    args = parser.parse_args()

    mesh_path = os.path.abspath(args.mesh)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(mesh_path):
        print(f"エラー: メッシュが見つかりません: {mesh_path}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    if args.from_files:
        # run_from_files: 出力ファイルは output_dir / {mesh_name}_optcuts.obj など
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.join(output_dir, f"{mesh_name}_optcuts.obj")
        print(f"run_from_files: {mesh_path} -> {output_path}")
        UV, F, n_iter = optcuts.run_from_files(
            mesh_path,
            output_path,
            max_iter=args.max_iter,
            lambda_init=args.lambda_init,
            scaffolding=not args.no_scaffolding,
            sigma1_bound=args.sigma1,
            sigma2_bound=args.sigma2,
            max_topo_iters=args.max_topo_iters,
            upper_bound=args.upper_bound,
        )
        print(f"反復: {n_iter}, 保存: {output_path}")
        return

    # run_main: main.cpp と同様の argv を組み立てる（batch.py 相当: 10 mesh 0.99 1 0 4.1 1 1.0 1.0 0）
    if args.headless:
        prog_mode = "100"
    elif args.offline_mode:
        prog_mode = "10"
    else:
        prog_mode = "0"
    argv_list = [
        "optcuts",
        prog_mode,
        mesh_path,
        str(args.lambda_init),
        str(args.test_id),
        str(args.method),
        str(args.upper_bound),
        "1" if not args.no_scaffolding else "0",
        str(args.sigma1),
        str(args.sigma2),
        str(args.init_cut),
    ]
    optcuts.set_output_folder(output_dir + os.sep)
    print("run_main argv:", argv_list)
    ret = optcuts.run_main(argv_list)
    if ret != 0:
        print(f"run_main が 0 以外で終了: {ret}")
        sys.exit(ret)
    UV, F = optcuts.get_result_mesh()
    print(f"反復: {optcuts.get_iter_num()}, 結果: UV {UV.shape}, F {F.shape}")


if __name__ == "__main__":
    main()
