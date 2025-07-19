import os
import sys
import shutil


def merge_logs_and_check(folder_path, merged_log_name="merged.log"):
    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".log")]
    if not log_files:
        print("No .log files found.")
        return

    merged_log_path = os.path.join(folder_path, merged_log_name)

    failure_patterns = ["out of memory", "error", "failed", "exception"]
    failures_found = []

    print(f"Found {len(log_files)} log files. Merging and checking...")

    with open(merged_log_path, "w", encoding="utf-8") as merged_file:
        for file in log_files:
            with open(file, "r", encoding="utf-8") as f:
                contents = f.read()
                merged_file.write(f"\n===== {os.path.basename(file)} =====\n")
                merged_file.write(contents)
                merged_file.write("\n")

                for pattern in failure_patterns:
                    if pattern.lower() in contents.lower():
                        failures_found.append((os.path.basename(file), pattern))

    print(f"✅ Merged log saved to: {merged_log_path}")

    # Move files to archive
    archive_folder = os.path.join(folder_path, "archive")
    os.makedirs(archive_folder, exist_ok=True)

    for file in log_files:
        shutil.move(file, os.path.join(archive_folder, os.path.basename(file)))

    print(f"📁 All log files moved to: {archive_folder}")

    if failures_found:
        print("\n🚨 Failures detected:")
        for fname, reason in failures_found:
            print(f"- {fname}: {reason}")
    else:
        print("\n✅ No failures detected in logs.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_logs_and_check.py <folder_path> [merged_log_name]")
        sys.exit(1)

    folder = sys.argv[1]
    merged_name = sys.argv[2] if len(sys.argv) > 2 else "merged.log"

    merge_logs_and_check(folder, merged_name)
