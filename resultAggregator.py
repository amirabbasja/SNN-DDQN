import os
import re
import sys
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
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)