from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from update_manager import (
    DEFAULT_LIMIT,
    get_status_payload,
    is_update_due,
    load_config,
    load_status,
    mark_update_failed,
    mark_update_started,
    run_incremental_update,
    save_status,
    validate_limit,
)

PROJECT_ROOT = Path(__file__).resolve().parent


class UpdateCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start_update(self, limit: int | None = None, trigger: str = "manual") -> tuple[bool, str]:
        normalized_limit = validate_limit(limit if limit is not None else load_config().get("crawl_limit", DEFAULT_LIMIT))
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "已有更新任务正在执行。"

            mark_update_started(normalized_limit, trigger)
            thread = threading.Thread(
                target=self._run_update,
                args=(normalized_limit, trigger),
                daemon=True,
                name=f"update-{trigger}",
            )
            self._thread = thread
            thread.start()
            return True, "更新任务已启动。"

    def _run_update(self, limit: int, trigger: str) -> None:
        try:
            run_incremental_update(limit=limit, trigger=trigger)
        except Exception as exc:
            mark_update_failed(limit, trigger, str(exc))
        finally:
            with self._lock:
                self._thread = None


UPDATE_COORDINATOR = UpdateCoordinator()


def reconcile_stale_running_status() -> None:
    status = load_status()
    if status.get("state") == "running" and not UPDATE_COORDINATOR.is_running():
        save_status(
            {
                "state": "idle",
                "message": "已重置上次未结束的更新状态。",
                "last_error": "",
            }
        )


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._handle_status()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/update":
            self._handle_update()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {format % args}")

    def _read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status_code: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_status(self) -> None:
        reconcile_stale_running_status()
        payload = get_status_payload()
        payload["runtime"] = {"is_running": UPDATE_COORDINATOR.is_running()}
        self._send_json(payload)

    def _handle_update(self) -> None:
        try:
            request_body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "请求体必须是合法 JSON。"}, status_code=HTTPStatus.BAD_REQUEST)
            return

        limit = request_body.get("limit")
        if limit is None:
            limit = load_config().get("crawl_limit", DEFAULT_LIMIT)

        try:
            raw_limit = int(limit)
        except (TypeError, ValueError):
            self._send_json({"error": "抓取数量必须是正整数。"}, status_code=HTTPStatus.BAD_REQUEST)
            return
        if raw_limit < 1 or raw_limit > 1000:
            self._send_json({"error": "抓取数量必须在 1 到 1000 之间。"}, status_code=HTTPStatus.BAD_REQUEST)
            return
        normalized_limit = validate_limit(raw_limit)

        started, message = UPDATE_COORDINATOR.start_update(normalized_limit, trigger="manual")
        payload = {
            "message": message,
            "limit": normalized_limit,
            "runtime": {"is_running": UPDATE_COORDINATOR.is_running()},
        }
        self._send_json(payload, status_code=HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)


def scheduler_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(60):
        reconcile_stale_running_status()
        if UPDATE_COORDINATOR.is_running():
            continue

        config = load_config()
        status = load_status()
        if is_update_due(config, status=status):
            started, message = UPDATE_COORDINATOR.start_update(
                config.get("crawl_limit", DEFAULT_LIMIT),
                trigger="weekly",
            )
            if not started:
                save_status({"message": message})


def run_server(host: str, port: int) -> None:
    reconcile_stale_running_status()
    stop_event = threading.Event()
    scheduler = threading.Thread(target=scheduler_loop, args=(stop_event,), daemon=True, name="weekly-scheduler")
    scheduler.start()

    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"服务已启动: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在关闭服务...")
    finally:
        stop_event.set()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Genealogy local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--once", action="store_true", help="run a single update job and exit")
    parser.add_argument("--limit", type=int, default=None, help="crawl limit for this run")
    parser.add_argument("--trigger", default="manual", help="update trigger name")
    args = parser.parse_args()

    if args.once:
        result = run_incremental_update(limit=args.limit, trigger=args.trigger)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
