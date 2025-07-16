from pathlib import Path

root_dir = Path("Instances/Processed")
output_txt = "mapf_jobs.txt"

heuristic_modes = ["No", "A", "B"]
objectives = ["makespan", "sum_of_costs"]

priority_filenames = [
    f"priority{i}-static.lp" if i <= 5 else f"priority{i}-kpath.lp"
    for i in range(1, 14)
]

wrapper_script = "scripts/run_clingo.py"

with open(output_txt, "w") as f:
    job_id = 0

    for map_folder in root_dir.iterdir():
        if not map_folder.is_dir():
            continue

        map_name = map_folder.name  # e.g. empty-32-32

        for scenario_folder in map_folder.iterdir():
            if not scenario_folder.is_dir():
                continue

            scen_name = scenario_folder.name

            if not (scen_name.endswith("0") or scen_name.endswith("1")):
                continue  # Only 0 or 1

            # Compose paths
            map_file = Path(f"Instances/maps/{map_name}.map")
            scen_file = Path(f"Instances/scenarios/{map_name}/{scen_name}.scen")

            if not map_file.exists():
                print(f"⚠️ Missing map file: {map_file}")
                continue

            if not scen_file.exists():
                print(f"⚠️ Missing scen file: {scen_file}")
                continue
            
            for objective in objectives:
                for heuristic in heuristic_modes:
                    if heuristic == "No":
                        cmd = (
                            f"/mnt/beegfs/home/shetty/.conda/envs/cmapf-env/bin/python {wrapper_script} "
                            f"--map_file {map_file.as_posix()} "
                            f"--scen_file {scen_file.as_posix()} "
                            f"--heuristic {heuristic} "
                            f"--objective {objective}"
                        )
                        f.write(cmd + "\n")
                        job_id += 1
                    else:
                        for prio_file in priority_filenames:
                            prio_path = scenario_folder / prio_file
                            if prio_path.exists():
                                cmd = (
                                    f"/mnt/beegfs/home/shetty/.conda/envs/cmapf-env/bin/python {wrapper_script} "
                                    f"--map_file {map_file.as_posix()} "
                                    f"--scen_file {scen_file.as_posix()} "
                                    f"--heuristic {heuristic} "
                                    f"--priority_file {prio_path.as_posix()} "
                                    f"--objective {objective}"
                                )
                                f.write(cmd + "\n")
                                job_id += 1
                            else:
                                print(f"⚠️ Missing: {prio_path}")

print(f"✅ Job file written: {output_txt}")
