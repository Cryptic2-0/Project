"""One-shot setup: create an AWS Budget at the project's monthly cost ceiling.

Run locally with valid AWS credentials:

    python scripts/setup_aws_budget.py --email you@example.com --limit 13

The budget fires email alerts at 80% (forecast) and 100% (actual). Idempotent:
re-running with the same `--name` updates the existing budget.

Why a script instead of clicking the console: the alarm survives an account
audit. The script is checked-in proof that the cost ceiling exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError


def _budget_payload(name: str, limit_usd: float) -> dict[str, Any]:
    return {
        "BudgetName": name,
        "BudgetLimit": {"Amount": f"{limit_usd:.2f}", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostTypes": {
            "IncludeTax": True,
            "IncludeSubscription": True,
            "UseBlended": False,
            "IncludeRefund": False,
            "IncludeCredit": False,
            "IncludeUpfront": True,
            "IncludeRecurring": True,
            "IncludeOtherSubscription": True,
            "IncludeSupport": True,
            "IncludeDiscount": True,
            "UseAmortized": False,
        },
    }


def _notifications(email: str) -> list[dict[str, Any]]:
    """80% forecasted + 100% actual. Both notify the same email."""
    subscribers = [{"SubscriptionType": "EMAIL", "Address": email}]
    return [
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80.0,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": subscribers,
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": subscribers,
        },
    ]


def create_or_update(account_id: str, name: str, limit_usd: float, email: str) -> str:
    client = boto3.client("budgets")
    budget = _budget_payload(name, limit_usd)
    notifications = _notifications(email)

    try:
        client.describe_budget(AccountId=account_id, BudgetName=name)
        client.update_budget(AccountId=account_id, NewBudget=budget)
        action = "updated"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NotFoundException":
            raise
        client.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=[
                {"Notification": n["Notification"], "Subscribers": n["Subscribers"]}
                for n in notifications
            ],
        )
        return "created"

    existing = client.describe_notifications_for_budget(AccountId=account_id, BudgetName=name)
    for existing_notif in existing.get("Notifications", []):
        client.delete_notification(
            AccountId=account_id, BudgetName=name, Notification=existing_notif
        )
    for n in notifications:
        client.create_notification(
            AccountId=account_id,
            BudgetName=name,
            Notification=n["Notification"],
            Subscribers=n["Subscribers"],
        )
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="moviesentiment-monthly", help="Budget name.")
    parser.add_argument(
        "--limit", type=float, default=13.0, help="Monthly USD ceiling (default: 13)."
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("MS_BUDGET_EMAIL"),
        help="Email for alerts. Falls back to MS_BUDGET_EMAIL env var.",
    )
    args = parser.parse_args()

    if not args.email:
        print("ERROR: --email or MS_BUDGET_EMAIL required.", file=sys.stderr)
        return 2

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    action = create_or_update(account_id, args.name, args.limit, args.email)
    print(f"Budget '{args.name}' {action}: ${args.limit:.2f}/mo -> {args.email}")
    print("  Alerts: 80% forecast, 100% actual")
    print(f"  Account: {account_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
