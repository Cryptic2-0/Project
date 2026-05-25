# mypy: ignore-errors
"""Locust load test for the MovieSentiment inference API.

Usage:
    locust -f scripts/load_test.py --host http://localhost:8000 \
           --headless -u 50 -r 5 -t 120s \
           --html docs/load_test_report.html
"""

from locust import HttpUser, between, task

_REVIEWS = [
    "An absolute masterpiece of modern cinema. The direction and acting are flawless.",
    "Completely unwatchable. Poor writing, wooden acting, and a plot that goes nowhere.",
    "One of the best films I have seen this year. Highly recommended.",
    "Boring and predictable. Save your money and skip this one.",
    "A decent film with some memorable moments, though the pacing drags in the middle.",
    "The cinematography is stunning but the story fails to deliver.",
    "Brilliant performances all round. A must-watch for any film lover.",
    "I walked out after 30 minutes. Simply dreadful.",
]


class SentimentUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_single(self) -> None:
        import random

        self.client.post(
            "/predict",
            json={"texts": [random.choice(_REVIEWS)]},
            name="/predict (single)",
        )

    @task(3)
    def predict_batch(self) -> None:
        import random

        batch = random.sample(_REVIEWS, k=min(4, len(_REVIEWS)))
        self.client.post(
            "/predict",
            json={"texts": batch},
            name="/predict (batch-4)",
        )

    @task(1)
    def healthz(self) -> None:
        self.client.get("/healthz", name="/healthz")
