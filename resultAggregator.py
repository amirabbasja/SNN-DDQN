import os
import re
import sys
import json
import zipfile


def parse_env_file(env_path):
    env = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    env[key] = val
    except FileNotFoundError:
        pass
    return env


def get_session_name(env_path):
    env = parse_env_file(env_path)
    session = env.get("session_name") or env.get("SESSION_NAME")
    if not session:
        raise RuntimeError(f"Missing 'session_name' in .env at {env_path}")
    return session


def list_run_dirs(runs_root):
    if not os.path.isdir(runs_root):
        raise RuntimeError(f"runs_data directory not found: {runs_root}")
    run_dirs = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p):
            run_dirs.append(p)
    run_dirs.sort()
    return run_dirs


def collect_target_files(run_dir):
    targets = []
    for root, _, files in os.walk(run_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".png", ".json"):
                targets.append(os.path.join(root, fn))
    return targets


def derive_run_folder_name(run_dir_name):
    m = re.search(r"\d+", run_dir_name)
    return m.group(0) if m else run_dir_name


def create_zip(session_name, runs_root, output_zip):
    run_dirs = list_run_dirs(runs_root)
    included_files = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run_dir in run_dirs:
            run_dir_name = os.path.basename(run_dir)
            run_folder_in_zip = derive_run_folder_name(run_dir_name)
            files = collect_target_files(run_dir)

            for fp in files:
                rel = os.path.relpath(fp, start=run_dir)
                arc = os.path.join(run_folder_in_zip, rel).replace("\\", "/")
                zf.write(fp, arcname=arc)
                included_files += 1

    return len(run_dirs), included_files


def get_telegram_credentials(env_path):
    env = parse_env_file(env_path)
    chat_id = env.get("telegram_chat_id")
    bot_token = env.get("telegram_bot_token")
    if not chat_id or not bot_token:
        raise RuntimeError("Missing 'telegram_chat_id' or 'telegram_bot_token' in .env")
    return chat_id, bot_token


def getMatchingRunConfigs(runs_dir="runs_data", current_algorithm=None):
    """
    Finds all run configurations in the specified directory that match the current algorithm.

    Args:
        runs_dir (str): The directory to search for run configurations. Defaults to "runs_data".
        current_algorithm (str, optional): The current algorithm to match. If None, attempts to read from "conf.json".

    Returns:
        list: A list of matching run configurations.
    """
    if current_algorithm is None:
        root_conf_path = os.path.join(".", "conf.json")
        if os.path.isfile(root_conf_path):
            try:
                with open(root_conf_path, "r", encoding="utf-8") as f:
                    root_conf = json.load(f)
                current_algorithm = root_conf.get("algorithm")
            except Exception:
                current_algorithm = None

    if not current_algorithm:
        return []

    if not os.path.isdir(runs_dir):
        return []

    matching_configs = []
    try:
        for item in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, item)
            if not os.path.isdir(run_path):
                continue

            conf_path = os.path.join(run_path, "conf.json")
            if not os.path.isfile(conf_path):
                candidate = None
                try:
                    for child in os.listdir(run_path):
                        child_conf = os.path.join(run_path, child, "conf.json")
                        if os.path.isfile(child_conf):
                            candidate = child_conf
                            break
                except Exception:
                    pass
                conf_path = candidate if candidate else None

            if conf_path and os.path.isfile(conf_path):
                try:
                    with open(conf_path, "r", encoding="utf-8") as f:
                        conf = json.load(f)
                    if conf.get("algorithm") == current_algorithm:
                        matching_configs.append(conf)
                except Exception:
                    continue
    except Exception:
        return matching_configs

    return matching_configs


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(project_root, ".env")
    runs_root = os.path.join(project_root, "runs_data")

    session_name = get_session_name(env_path)
    output_zip = os.path.join(project_root, f"{session_name}_runs_results.zip")

    num_runs, num_files = create_zip(session_name, runs_root, output_zip)

    # Upload to Telegram
    chat_id, bot_token = get_telegram_credentials(env_path)
    upload_to_telegram(output_zip, chat_id, bot_token)
    print(f"Uploaded zip to Telegram chat {chat_id}")
    print(f"Created zip: {output_zip}")
    print(f"Runs included: {num_runs}")
    print(f"Files included: {num_files}")


def upload_to_telegram(file_path, chat_id, bot_token):
    try:
        import requests  # imported lazily to avoid global dependency
    except ImportError:
        raise RuntimeError("The 'requests' package is required. Install via 'pip install requests'.")

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": (os.path.basename(file_path), f, "application/zip")}
        data = {"chat_id": str(chat_id)}
        resp = requests.post(url, data=data, files=files, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Telegram upload failed: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    try:
        if "--check_duplicate_config" in sys.argv: 
            try:
                conf = json.loads(sys.argv[sys.argv.index("--check_duplicate_config") + 1])
                # Remove the keys that may change and are unnecessary for comparing
                conf.pop("tuning", None)
                conf.pop("tune", None)
                conf.pop("debug", None)
                conf.pop("local_backup", None)
                conf.pop("upload_to_cloud", None)
                conf.pop("train_max_time", None)
                conf.pop("stop_learning_at_win_percent", None)
                conf.pop("max_run_time", None)
                conf.pop("extra_info", None)
                conf.pop("agents", None)
                conf.pop("continue_run", None)
                conf.pop("finished", None)

                # Get all configs that have the same algorithm (located in runs_data directory)
                similarConfigs = getMatchingRunConfigs(runs_dir="runs_data", current_algorithm=conf["algorithm"])

                found_duplicate = False
                for _conf in similarConfigs:
                    # See if training has finished
                    _finished = True if str(_conf.get("finished", None)) == "True" else False if str(_conf.get("finished", None)) == "False" else None

                    # Remove the keys that may change and are unnecessary for comparing
                    _conf.pop("tuning", None)
                    _conf.pop("tune", None)
                    _conf.pop("debug", None)
                    _conf.pop("local_backup", None)
                    _conf.pop("upload_to_cloud", None)
                    _conf.pop("train_max_time", None)
                    _conf.pop("stop_learning_at_win_percent", None)
                    _conf.pop("max_run_time", None)
                    _conf.pop("extra_info", None)
                    _conf.pop("agents", None)
                    _conf.pop("continue_run", None)
                    _conf.pop("finished", None)

                    if conf == _conf and _finished == True:
                        print("true: finished duplicate found")
                        found_duplicate = True
                        break
                    elif conf == _conf and _finished == False:
                        print("false: unfinished duplicate found")
                        found_duplicate = True
                        break

                if not found_duplicate:
                    print("false: no duplicates found")

            except Exception as e:
                raise Exception("Invalid JSON format in --check_duplicate_config parameter. Configuration should be passed exactly after --check_duplicate_config flag.")
        elif "--upload_to_telegram" in sys.argv:
            main()
        else:
            print("No valid flag found. Use --check_duplicate_config or --upload_to_telegram.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)