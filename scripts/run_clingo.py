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
parser.add_argument("--instance_folder", required=True, help="Path to folder containing .lp instances")
parser.add_argument("--priority_file", required=False, help="Path to priority.lp file")
parser.add_argument("--heuristic", required=True, choices=["No", "A", "B"], help="Heuristic strategy")

args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODING_DIR = os.path.join(PROJECT_ROOT, "scripts", "encodings")
FOLDER_PATH = os.path.join(PROJECT_ROOT, args.instance_folder)
PRIORITY_FILE = os.path.join(PROJECT_ROOT, args.priority_file) if args.priority_file else ""
MAPF_Solver = os.path.join(PROJECT_ROOT, "scripts", "MAPF_with_priority.py")
Heuristics = args.heuristic

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

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

folder_name = os.path.basename(os.path.normpath(FOLDER_PATH))

priority_suffix = ""
if PRIORITY_FILE:
    priority_suffix = "_" + os.path.basename(PRIORITY_FILE).replace(".lp", "")

EXCEL_FILE = os.path.join("results",f"{folder_name}_{Heuristics}{priority_suffix}.xlsx")
log_filename = os.path.join("logs", f"{folder_name}_{Heuristics}{priority_suffix}.log")

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
columns = ['Processing File', 'Heuristics', 'Load Time', 'Ground Instance Time', 'Extract Problem Time',
           'Reachable Time', 'Ground Encoding Time', 'Solve Encoding Time', 'Delta', 'Total Time']

# Load existing results from Excel file (if it exists)

if os.path.exists(EXCEL_FILE):
    results_df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
else:
    results_df = pd.DataFrame(columns=columns)

TIMEOUT = 300  # Timeout for a single iteration in seconds

# Decorator to log cumulative time and other details
def log_cumulative_time(func):
    @wraps(func)
    def wrapper(file_path, *args, **kwargs):
        cumulative_time = 0  # Initialize cumulative time
        delta = 0  # Start with delta = 0    
        logging.info(f"file : {os.path.basename(file_path)},Heuristic : {Heuristics}")
        while True:
            #start = timeit.default_timer()
            result, time_spent,stats  = func(file_path,delta, cumulative_time,*args, **kwargs)  # Call the decorated function
            #end = timeit.default_timer()
            #iteration_time = end - start
            cumulative_time += time_spent
            
            
            # Append the results for this delta iteration to the DataFrame
            results_df.loc[len(results_df)] = [
                os.path.basename(file_path), Heuristics,
                float(stats.get('Load', 0)) if stats.get('Load') else np.nan,
                float(stats.get('Ground Instance', 0)) if stats.get('Ground Instance') else np.nan,
                float(stats.get('Extract Problem', 0)) if stats.get('Extract Problem') else np.nan,
                float(stats.get('Reachable', 0)) if stats.get('Reachable') else np.nan,
                float(stats.get('Ground Encoding', 0)) if stats.get('Ground Encoding') else np.nan,
                float(stats.get('Solve Encoding', 0)) if stats.get('Solve Encoding') else np.nan,
                delta, float(cumulative_time)  # Ensure numeric values
                ] 

            if result == "timeout":
                logging.info(f" delta : {delta}, time : {cumulative_time:.2f} (timeout)")
                return True    # Stop processing further files
            elif result == "solution_found":
                logging.info(f"delta : {delta}, time : {cumulative_time:.2f} (solution found)")
                return False    # Continue processing next file

            logging.info(f" delta : {delta}, time_spent : {time_spent:.2f}")
            delta += 1  # Increment delta for the next iteration
    return wrapper

# Main function to run the solver
@log_cumulative_time
def run_solver(file_path,delta, cumulative_time,*args, **kwargs):
    python_executable = sys.executable  # Detect current Python executable
    command = [
        python_executable,
        MAPF_Solver,
        f"--delta={delta}",
        f"--heuristic-strategy={Heuristics}",
        encoding_base,  
    ]
    if heuristic_lp:
        command.append(heuristic_lp)

    if Heuristics != "No":
        command.append(PRIORITY_FILE)

    command.append(file_path)  # the instance file

    if Heuristics != "No":
        command.append("--heuristic=domain")
        command.append("--opt-strategy=bb") #usc

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
  
    stats = {}
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

def extract_number(filename):
    """Extract the trailing number from the filename for sorting."""
    match = re.search(r'_(\d+)(?:\.lp)?$', filename)
    return int(match.group(1)) if match else float('inf')


# Get all .lp files in the folder except priority files and process them
def process_all_files():
    lp_files = sorted(
    [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith(".lp") and not f.startswith("priority")],
    key=lambda f: extract_number(os.path.basename(f)))  # Sort by extracted number

    if not lp_files:
        logging.warning(f"No instance .lp files found in {FOLDER_PATH}")
        return

    logging.info(f"Processing folder: {FOLDER_PATH.split(os.sep)[-1]} with {Heuristics} Heuristics")  # Log the folder being processed

    for file_path in lp_files:

        timeout_occurred = run_solver(file_path)  # Process each file
        if timeout_occurred:  # Stop further processing if timeout occurred
            break

    results_df.to_excel(EXCEL_FILE, index=False)

# Start processing all files
if __name__ == "__main__":
    process_all_files()
    logging.info(f"Finished processing. Results written to {EXCEL_FILE}")