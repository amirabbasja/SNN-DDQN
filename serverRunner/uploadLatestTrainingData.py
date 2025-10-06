import os
import re
import sys
import time
import json
import typing as t
import requests


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEN_MB = 10 * 1024 * 1024


def project_root() -> str:
    # serverRunner sits under the project root; go one level up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_runs_dir() -> str:
    return os.path.join(project_root(), "runs_data")


def load_env_vars_from_file(env_path: str) -> dict:
    data = {}
    if not os.path.isfile(env_path):
        return data
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    data[key] = val
    except Exception:
        pass
    return data


def resolve_credentials(
    bot_token_arg: t.Optional[str],
    chat_id_arg: t.Optional[str],
) -> t.Tuple[str, str]:
    # Priority: CLI args -> environment -> .env file
    if bot_token_arg and chat_id_arg:
        return bot_token_arg, chat_id_arg

    env = os.environ
    bot_token = bot_token_arg or env.get("TELEGRAM_BOT_TOKEN") or env.get("telegram_bot_token")
    chat_id = chat_id_arg or env.get("TELEGRAM_CHAT_ID") or env.get("telegram_chat_id")

    if bot_token and chat_id:
        return bot_token, chat_id

    # Try reading .env at project root
    env_file = os.path.join(project_root(), ".env")
    loaded = load_env_vars_from_file(env_file)
    bot_token = bot_token or loaded.get("TELEGRAM_BOT_TOKEN") or loaded.get("telegram_bot_token")
    chat_id = chat_id or loaded.get("TELEGRAM_CHAT_ID") or loaded.get("telegram_chat_id")

    if not bot_token or not chat_id:
        raise ValueError(
            "Missing Telegram credentials. Provide --bot-token and --chat-id, "
            "or set environment variables TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID."
        )
    return bot_token, chat_id


def find_latest_numeric_dir(runs_dir: str) -> t.Optional[str]:
    if not os.path.isdir(runs_dir):
        return None
    candidates: t.List[t.Tuple[int, str]] = []
    for name in os.listdir(runs_dir):
        full = os.path.join(runs_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"\d+", name):
            try:
                candidates.append((int(name), full))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def list_images_recursive(dir_path: str) -> t.List[str]:
    results: t.List[str] = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                results.append(os.path.join(root, fn))
    # Sort by filename then path for stable ordering
    results.sort(key=lambda p: (os.path.basename(p).lower(), p.lower()))
    return results


def _api_url(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def _handle_rate_limit(resp_json: dict, default_sleep: float = 1.5) -> float:
    # Telegram may return parameters.retry_after
    retry_after = 0.0
    try:
        params = resp_json.get("parameters") or {}
        ra = params.get("retry_after")
        if isinstance(ra, (int, float)) and ra > 0:
            retry_after = float(ra)
    except Exception:
        pass
    return retry_after if retry_after > 0 else default_sleep


def send_photo_with_retry(
    bot_token: str,
    chat_id: str,
    image_path: str,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> bool:
    # If file is too large for sendPhoto, fallback to sendDocument
    method = "sendPhoto"
    field = "photo"
    size = os.path.getsize(image_path)
    if size > TEN_MB:
        method = "sendDocument"
        field = "document"

    url = _api_url(bot_token, method)
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        try:
            with open(image_path, "rb") as f:
                files = {field: (os.path.basename(image_path), f)}
                data = {"chat_id": chat_id, "caption": os.path.basename(image_path)}
                resp = requests.post(url, data=data, files=files, timeout=timeout)
            if resp.status_code == 200:
                rj = resp.json()
                if rj.get("ok"):
                    return True
                # Non-ok response, maybe rate limit
                sleep_s = _handle_rate_limit(rj)
                time.sleep(sleep_s)
            elif resp.status_code == 429:
                try:
                    rj = resp.json()
                except Exception:
                    rj = {}
                sleep_s = _handle_rate_limit(rj)
                time.sleep(sleep_s)
            else:
                # Other HTTP errors: small backoff
                time.sleep(1.0)
        except requests.RequestException:
            time.sleep(1.5)
        except Exception:
            # Unexpected error; do not retry infinitely
            break
    return False


def send_media_group_with_retry(
    bot_token: str,
    chat_id: str,
    image_paths: t.List[str],
    timeout: float = 60.0,
    max_retries: int = 3,
) -> bool:
    # Sends up to 10 images as a single album using sendMediaGroup.
    # Skips files over 10MB (they should be sent as documents separately).
    batch = [p for p in image_paths if os.path.getsize(p) <= TEN_MB][:10]
    if not batch:
        return False

    url = _api_url(bot_token, "sendMediaGroup")
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        files = {}
        media = []
        try:
            # Prepare multipart attachments
            for idx, path in enumerate(batch):
                attach_key = f"file{idx}"
                files[attach_key] = (os.path.basename(path), open(path, "rb"))
                media.append({
                    "type": "photo",
                    "media": f"attach://{attach_key}",
                    "caption": os.path.basename(path),
                })

            data = {
                "chat_id": chat_id,
                "media": json.dumps(media),
            }

            resp = requests.post(url, data=data, files=files, timeout=timeout)
            if resp.status_code == 200:
                rj = resp.json()
                # ok:true means the whole album was accepted
                if rj.get("ok"):
                    return True
                sleep_s = _handle_rate_limit(rj)
                time.sleep(sleep_s)
            elif resp.status_code == 429:
                try:
                    rj = resp.json()
                except Exception:
                    rj = {}
                sleep_s = _handle_rate_limit(rj)
                time.sleep(sleep_s)
            else:
                time.sleep(1.0)
        except requests.RequestException:
            time.sleep(1.5)
        except Exception:
            break
        finally:
            # Ensure file handles are closed
            for v in files.values():
                try:
                    v[1].close()
                except Exception:
                    pass
    return False


def main(argv: t.List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload images from a run directory to a Telegram chat as albums."
    )
    parser.add_argument("--bot-token", type=str, help="Telegram bot token")
    parser.add_argument("--chat-id", type=str, help="Telegram chat ID")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=default_runs_dir(),
        help="Path to runs_data directory (default: project_root/runs_data)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay between uploads in seconds to avoid rate limits",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        help="Specific numeric run directory to use (defaults to latest if omitted)",
    )
    args = parser.parse_args(argv)

    try:
        bot_token, chat_id = resolve_credentials(args.bot_token, args.chat_id)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2

    runs_dir = args.runs_dir
    target_dir = None
    if args.run_number is not None:
        candidate = os.path.join(runs_dir, str(args.run_number))
        if os.path.isdir(candidate):
            target_dir = candidate
        else:
            print(f"[ERROR] Run directory not found: {candidate}")
            return 3
    else:
        latest_dir = find_latest_numeric_dir(runs_dir)
        if not latest_dir:
            print(f"[ERROR] No numeric directories found in: {runs_dir}")
            return 3
        target_dir = latest_dir

    images = list_images_recursive(target_dir)
    if not images:
        print(f"[WARN] No images found in: {target_dir}")
        return 0

    # Partition into <=10MB (album-capable) and >10MB (document-only)
    small_images = [p for p in images if os.path.getsize(p) <= TEN_MB]
    large_images = [p for p in images if os.path.getsize(p) > TEN_MB]

    print(f"[INFO] Uploading {len(images)} images from: {target_dir}")
    print(f"[INFO] Album-eligible (<=10MB): {len(small_images)}, large (>10MB): {len(large_images)}")

    ok = 0
    failed = 0

    # Send albums in batches of up to 10
    for start in range(0, len(small_images), 10):
        batch = small_images[start:start + 10]
        success = send_media_group_with_retry(bot_token, chat_id, batch)
        status = "OK" if success else "FAIL"
        batch_names = [os.path.relpath(p, target_dir) for p in batch]
        print(f"[ALBUM {start//10 + 1}] {status} - {batch_names}")
        if success:
            ok += len(batch)
        else:
            failed += len(batch)
        time.sleep(max(0.0, args.delay))

    # Send large images individually as documents
    for idx, img in enumerate(large_images, start=1):
        success = send_photo_with_retry(bot_token, chat_id, img)
        status = "OK" if success else "FAIL"
        print(f"[LARGE {idx}/{len(large_images)}] {status} - {os.path.relpath(img, target_dir)}")
        if success:
            ok += 1
        else:
            failed += 1
        time.sleep(max(0.0, args.delay))

    print(f"[DONE] Sent: {ok}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))