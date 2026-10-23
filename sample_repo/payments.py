"""Payment workflow for the bundled demonstration repository.

These functions are examples, not an integration with a payment provider.
"""
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class PaymentAttempt:
    order_id: str
    amount: float
    attempts: int = 0
    status: str = 'pending'


def retry_payment(payment: PaymentAttempt, max_attempts: int = 3) -> PaymentAttempt:
    """Retry a failed payment until its retry budget is exhausted.

    Only failed attempts are eligible. The caller must send the returned
    pending attempt to a payment provider; this helper never charges a card.
    """
    if payment.status != 'failed':
        raise ValueError('Only failed payments can be retried')
    if payment.attempts >= max_attempts:
        raise ValueError('Payment retry limit reached')
    return PaymentAttempt(
        order_id=payment.order_id,
        amount=payment.amount,
        attempts=payment.attempts + 1,
        status='pending',
    )


def calculate_refund(paid_amount: float, returned_fraction: float) -> float:
    """Calculate a proportional refund for a partially returned order."""
    if paid_amount < 0 or not 0 <= returned_fraction <= 1:
        raise ValueError('Invalid refund amount or fraction')
    return round(paid_amount * returned_fraction, 2)


def payment_receipt(payment: PaymentAttempt) -> dict:
    """Create receipt metadata after a successful payment."""
    if payment.status != 'paid':
        raise ValueError('Payment is not complete')
    return {'order_id': payment.order_id, 'amount': payment.amount,
            'issued_at': datetime.now(timezone.utc).isoformat()}


def idempotency_key(order_id: str, attempt: int) -> str:
    """Generate a stable request key to identify duplicate payment attempts."""
    if attempt < 0:
        raise ValueError('Attempt cannot be negative')
    return f'payment:{order_id}:{attempt}'
