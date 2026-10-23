"""Shipping quote and delivery helpers."""
from datetime import date, timedelta


def shipping_cost(weight_kg: float, express: bool = False) -> float:
    """Calculate delivery charges from parcel weight and shipping speed."""
    if weight_kg <= 0:
        raise ValueError('Weight must be positive')
    base = 4.0 + weight_kg * 1.5
    return round(base * (1.8 if express else 1), 2)


def delivery_date(dispatch: date, business_days: int = 3) -> date:
    """Estimate arrival while excluding weekends from delivery days."""
    if business_days < 0:
        raise ValueError('Days cannot be negative')
    arrival = dispatch
    while business_days:
        arrival += timedelta(days=1)
        if arrival.weekday() < 5:
            business_days -= 1
    return arrival


def valid_postcode(postcode: str) -> bool:
    """Check whether an Indian postal PIN has six digits and a nonzero prefix."""
    return postcode.isascii() and postcode.isdigit() and len(postcode) == 6 and postcode[0] != '0'
