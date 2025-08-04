import subprocess
import sys
import timeit
import logging
from functools import wraps
import os
import re
import pandas as pd
import openpyxl 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--priority_file", required=False, help="Path to priority.lp file")
parser.add_argument("--heuristic", required=True, choices=["No", "A", "B"], help="Heuristic strategy")
parser.add_argument("--map_file", required=True, help="Path to the .map file")
parser.add_argument("--scen_file", required=True, help="Path to the .scen file")
parser.add_argument("--objective", required=True, choices=["makespan", "sum_of_costs"], help="Optimization objective")
parser.add_argument("--noreach", action="store_true", help="Use encoding without reachability")

args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODING_DIR = os.path.join(PROJECT_ROOT, "scripts", "encodings")
PRIORITY_FILE = os.path.join(PROJECT_ROOT, args.priority_file) if args.priority_file else ""
MAPF_Solver = os.path.join(PROJECT_ROOT, "scripts", "MAPF_with_priority.py")

Heuristics = args.heuristic
map_file = args.map_file
scen_file = args.scen_file
objective = args.objective

if args.noreach:
    encoding_base = os.path.join(ENCODING_DIR, "encoding_no_reach.lp")
else:
    encoding_base = os.path.join(ENCODING_DIR, "encoding_base.lp")
if Heuristics == "No":
    heuristic_lp = None
elif Heuristics == "A":
    heuristic_lp = os.path.join(ENCODING_DIR, "heuristics_a.lp")
elif Heuristics == "B":
    heuristic_lp = os.path.join(ENCODING_DIR, "heuristics_b.lp")
else:
    raise ValueError("Unsupported heuristic type")

if Heuristics != "No" and not args.priority_file:
    raise ValueError("A priority file must be provided when using heuristics A or B.")

# Set locale to use comma as a decimal separator
#locale.setlocale(locale.LC_NUMERIC, "de_DE")

scen_name = os.path.splitext(os.path.basename(scen_file))[0]

priority_suffix = ""
if PRIORITY_FILE:
    priority_suffix = "_" + os.path.basename(PRIORITY_FILE).replace(".lp", "")

# Add subfolders based on objective
logs_dir = os.path.join("logs", objective)
results_dir = os.path.join("results", objective)

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

EXCEL_FILE = os.path.join(results_dir, f"{scen_name}_{Heuristics}{priority_suffix}.xlsx")
log_filename = os.path.join(logs_dir, f"{scen_name}_{Heuristics}{priority_suffix}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
   
# Initialize a DataFrame to hold the results (this will be saved to Excel later)
columns = ['Instance', 'Agents', 'Heuristics', 'Load Time', 'Ground Instance Time', 'Extract Problem Time',
           'Reachable Time', 'Ground Encoding Time', 'Solve Encoding Time', 'Delta/Horizon', 'Total Time',
           'Domain Choices']

# Load existing results from Excel file (if it exists)

if os.path.exists(EXCEL_FILE):
    results_df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
else:
    results_df = pd.DataFrame(columns=columns)

TIMEOUT = 300  # Timeout for a single iteration in seconds

# Decorator to log cumulative time and other details
def log_cumulative_time(func):
    @wraps(func)
    def wrapper(agent_count, *args, **kwargs):
        cumulative_time = 0  # Initialize cumulative time
        if objective == "makespan":
            delta = compute_min_horizon(agent_count)
        else:  # objective == "sum_of_costs"
            delta = 0

        logging.info(f"Agents: {agent_count}, Heuristic: {Heuristics}")
        while True:
            #start = timeit.default_timer()
            priority_file_to_use = PRIORITY_FILE

            if Heuristics in ["A", "B"] and PRIORITY_FILE:
                base, fname = os.path.split(PRIORITY_FILE)
                if "kpath" in fname:  # metrics 6–13
                    # Insert agent_count folder before filename
                    base = os.path.join(base, str(agent_count))
                    priority_file_to_use = os.path.join(base, fname)

            result, time_spent,stats  = func(agent_count, delta, cumulative_time, priority_file_to_use,*args, **kwargs)  # Call the decorated function
            #end = timeit.default_timer()
            #iteration_time = end - start
            cumulative_time += time_spent
            
            instance_id = os.path.splitext(os.path.basename(scen_file))[0]
            heu = Heuristics + priority_suffix.split("-")[0]
            # Append the results for this delta iteration to the DataFrame
            results_df.loc[len(results_df)] = [
                instance_id, agent_count, heu,
                float(stats.get('Load', np.nan)),
                float(stats.get('Ground Instance', np.nan)),
                float(stats.get('Extract Problem', np.nan)),
                float(stats.get('Reachable', np.nan)),
                float(stats.get('Ground Encoding', np.nan)),
                float(stats.get('Solve Encoding', np.nan)),
                delta,
                float(cumulative_time),
                stats.get('Domain Choices', np.nan)
                ]

            if result == "timeout":
                logging.info(f" delta/horizon : {delta}, time : {cumulative_time:.2f} (timeout)")
                return True    # Stop processing further files
            elif result == "solution_found":
                logging.info(f"delta/horizon : {delta}, time : {cumulative_time:.2f} (solution found)")
                return False    # Continue processing next file

            logging.info(f" delta/horizon : {delta}, time_spent : {time_spent:.2f}")
            delta += 1  # Increment delta for the next iteration
    return wrapper

# Main function to run the solver
@log_cumulative_time
def run_solver(agent_count,delta = 0, cumulative_time = 0, priority_file="", *args, **kwargs):
    python_executable = sys.executable  # Detect current Python executable
    
    command = [
        python_executable,
        MAPF_Solver,
        f"--heuristic-strategy={Heuristics}",
        f"--map-file={map_file}",
        f"--scen-file={scen_file}",
        f"--agents={agent_count}",
        encoding_base,
    ]
    if heuristic_lp:
        command.append(heuristic_lp)

    if Heuristics != "No" and priority_file:
        if not os.path.exists(priority_file):
            logging.warning(f"Priority file not found: {priority_file}")
        command.append(priority_file)
        command.append("--heuristic=domain")

    if objective == "makespan":
        command.append(f"--horizon={delta}")
    else:
        command.append(f"--delta={delta}")

    command.append("--single-shot")
    command.append("--stats")


    logging.info("Running command: " + " ".join(command))

    start_time = timeit.default_timer()
    try:
        remaining_time = max(0.1, TIMEOUT - cumulative_time)
        result = subprocess.run(command, capture_output=True, text=True, timeout=remaining_time)
    except subprocess.TimeoutExpired:
        elapsed_time = timeit.default_timer() - start_time  # Time spent before timeout
        logging.info(f"Timeout occurred after {elapsed_time:.2f} seconds")
        return "timeout", elapsed_time,{}  # Return timeout and time spent so far
       


    iteration_time = timeit.default_timer() - start_time  # Time taken for this iteration
    # Capture the clingo statistics
    stdout = result.stdout
    print(stdout)  # Print the output for debugging
    stats = {}
    # Extract number of domain choices
    domain_match = re.search(r"Choices\s+:\s+\d+\s+\(Domain:\s+(\d+)\)", stdout)
    if domain_match:
        stats["Domain Choices"] = int(domain_match.group(1))
    else:
        stats["Domain Choices"] = np.nan  
    
    

    # Extract and log time statistics if available

    stats_section = re.search(r"Statistics:(.*?)(?:SATISFIABLE|UNSATISFIABLE|Finish)", stdout, re.DOTALL)# or re.search(r"Statistics:(.*?)UNSATISFIABLE", stdout, re.DOTALL)
    
    if stats_section:
        # Extract and log the time stats directly
        for line in stats_section.group(1).splitlines():
            if any(line.startswith(prefix) for prefix in ["Load", "Ground Instance", "Extract Problem", "Reachable", "Ground Encoding", "Solve Encoding"]):
                category, time_value = line.split(":")
                stats[category] = time_value.strip()
                #logging.info(f"{category} Time: {time_value} seconds")

    if "The problem is satisfiable!" in stdout:
        return "solution_found", iteration_time, stats

    return "continue", iteration_time, stats


def compute_min_horizon(agent_count: int) -> int:
    python_executable = sys.executable
    command = [
        python_executable,
        MAPF_Solver,
        f"--map-file={map_file}",
        f"--scen-file={scen_file}",
        f"--agents={agent_count}",
        "--compute-min-horizon"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    stdout = result.stdout

    for line in stdout.splitlines():
        if "Min Horizon" in line:
            return int(line.strip().split()[-1])

    raise RuntimeError("Failed to determine Min Horizon from solver output")


# Get all .lp files in the folder except priority files and process them
def process_all_files():
    logging.info(f"Processing map: {map_file}, scenario: {scen_file} with {Heuristics} Heuristics")
    max_agents = 100 #get_max_agents_from_scen(scen_file)

    agent_count = 5  # starting agents
    step = 5         # increment step
    
    while agent_count <= max_agents:
        logging.info(f"Running with {agent_count} agents")

        timeout_occurred = run_solver(agent_count)
        if timeout_occurred:  # stop further processing if timeout occurred
            break

        agent_count += step

    results_df.to_excel(EXCEL_FILE, index=False)
    

# Start processing all files
if __name__ == "__main__":
    process_all_files()
    logging.info(f"Finished processing. Results written to {EXCEL_FILE}")