"""Locust load profile for the streamed/JSON `POST /ask` endpoint (MLOPS_TODO #74, #75).

Two profiles, both UTF-8/RTL safe:

  smoke  (low concurrency, sanity):
      uv run locust -f tests/load/locustfile.py --headless -u 5 -r 1 -t 1m \
          --host http://localhost:8000 --html reports/load/smoke.html

  report (target hardware, M2.5 reranker config, 50 users):
      uv run locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 5m \
          --host http://localhost:8000 --html reports/load/report.html

Environment:
  RAG_API_PREFIX  route prefix, default `/api/v1`
  RAG_API_KEY     when API_KEY_ENABLED, sent as `Authorization: Bearer <key>`
  RAG_RATE_KEY    per-user `X-Rate-Key` so buckets are not shared by egress IP

The profile fails the run on upstream errors (`HTTP >= 400` and any
`event: error` frame) and reports the abstention rate separately, so a run
cannot pass by refusing every question.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

API_PREFIX = os.getenv("RAG_API_PREFIX", "/api/v1")
ASK_PATH = f"{API_PREFIX}/ask"
API_KEY = os.getenv("RAG_API_KEY")

QUESTIONS: list[str] = [
    "ما هي آثار عقد البيع في القانون المدني المصري؟",
    "متى تسقط الدعوى المدنية بالتقادم؟",
    "ما هي شروط صحة الإيجار؟",
    "كيف يتم إثبات الالتزام؟",
    "ما حكم الشرط الجزائي في العقد؟",
    "ما هي أسباب انقضاء الالتزام؟",
    "ما الفرق بين الفسخ والانفساخ؟",
    "ما هي مسؤولية المتبوع عن أعمال التابع؟",
    "متى يكون العقد باطلاً بطلاناً مطلقاً؟",
    "ما هي آثار الإثراء بلا سبب؟",
    "ما هي أحكام التعويض عن الضرر المعنوي؟",
    "كيف تنقضي الملكية؟",
    "ما هي ضمانات عيوب المبيع؟",
    "ما حكم التصرف في ملك الغير؟",
]

ABSTAINED_COUNT = 0
ANSWERED_COUNT = 0
ERROR_COUNT = 0


def _headers(user: "AskUser") -> dict[str, str]:
    headers = {"Accept": "application/json", "X-Rate-Key": user.rate_key}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


class AskUser(HttpUser):
    """Streams a single question per task and validates the RAG contract."""

    wait_time = between(1, 3)
    request_timeout = 50  # seconds

    def on_start(self) -> None:
        self.rate_key = f"locust-{id(self)}"
        if API_KEY:
            self.client.headers.update({"Authorization": f"Bearer {API_KEY}"})

    @task
    def ask(self) -> None:
        global ABSTAINED_COUNT, ANSWERED_COUNT, ERROR_COUNT

        question = random.choice(QUESTIONS)
        with self.client.post(
            ASK_PATH,
            json={"question": question},
            headers=_headers(self),
            catch_response=True,
            name="/ask",
            timeout=self.request_timeout,
        ) as response:
            if "event: error" in response.text:
                ERROR_COUNT += 1
                response.failure("upstream event: error frame")
                return

            if response.status_code == 422:
                response.failure("422 validation error (empty/short question)")
                return

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("non-JSON response body")
                return

            if not isinstance(body, dict):
                response.failure("unexpected response shape")
                return

            if body.get("abstained") is True:
                if not body.get("reason"):
                    response.failure("abstention without reason")
                    return
                ABSTAINED_COUNT += 1
                response.success()
                return

            if not body.get("answer"):
                response.failure("answer missing or empty")
                return

            if not body.get("sources"):
                response.failure("answered without resolvable citations")
                return

            ANSWERED_COUNT += 1
            response.success()


@events.quitting.add_listener
def report_abstention(environment, **kwargs) -> None:
    total = ABSTAINED_COUNT + ANSWERED_COUNT
    if total == 0:
        print("[locust] no responses recorded; aborting without a verdict")
        environment.process_exit_code = 1
        return

    rate = ABSTAINED_COUNT / total
    print(
        f"[locust] abstention rate {rate:.2%} "
        f"({ABSTAINED_COUNT}/{total}), answered {ANSWERED_COUNT}, "
        f"errors {ERROR_COUNT}"
    )

    if ANSWERED_COUNT == 0:
        print("[locust] run refused every question; failing the profile")
        environment.process_exit_code = 1
