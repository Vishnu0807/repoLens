"""Small order workflow used to demonstrate RepoLens."""

from dataclasses import dataclass

from validators import check_quantity, valid_email


@dataclass
class Order:
    customer_email: str
    unit_price: float
    quantity: int

    def total(self) -> float:
        """Calculate the total price of this order."""
        return round(self.unit_price * self.quantity, 2)


def apply_discount(amount: float, percent: float) -> float:
    """Reduce an amount by a percentage discount."""
    if not 0 <= percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    return round(amount * (1 - percent / 100), 2)


def create_order(email: str, price: float, quantity: int) -> Order:
    """Validate customer input and construct an order."""
    if not valid_email(email):
        raise ValueError("Invalid customer email")
    check_quantity(quantity)
    if price < 0:
        raise ValueError("Price cannot be negative")
    return Order(email, price, quantity)


def checkout(order: Order, discount: float = 0) -> float:
    """Compute the final amount due for an order."""
    return apply_discount(order.total(), discount)

