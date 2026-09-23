# Runs planet_bench.gd for every view x render path in fresh processes, writes bench_results.json.
#   python run_bench.py [godot_exe] [out_dir]
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
GODOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, r"..\..\..\..\..\bin\godot.windows.editor.x86_64.console.exe")
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bench_out"))
os.makedirs(OUT, exist_ok=True)

results = []
for view in ("orbit", "surface"):
    for mode in (0, 1):
        t0 = time.time()
        p = subprocess.run([GODOT, "--path", HERE, "--script", "planet_bench.gd", "--resolution", "1280x720",
                            "--", view, str(mode), OUT], capture_output=True, text=True, errors="replace")
        log = p.stdout + p.stderr
        open(os.path.join(OUT, f"log_{view}_{mode}.txt"), "w").write(log)
        line = next((l for l in log.splitlines() if l.startswith("BENCH_JSON=")), None)
        if line is None:
            sys.exit(f"{view}/{mode}: no result (exit {p.returncode}), see log_{view}_{mode}.txt")
        r = json.loads(line[len("BENCH_JSON="):])
        r["process_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(view, r["path"], "load", r["load"]["load_s"], "s settled", r["load"]["settled"],
              "fps", r["steady"]["avg_fps"], flush=True)

json.dump(results, open(os.path.join(OUT, "bench_results.json"), "w"), indent=1)
