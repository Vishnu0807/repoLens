"""Small reporting functions for completed orders."""
from collections import Counter


def total_revenue(orders: list[dict]) -> float:
    """Sum revenue from paid orders, excluding cancelled and unpaid orders."""
    return round(sum(order['amount'] for order in orders if order['status'] == 'paid'), 2)


def average_order_value(orders: list[dict]) -> float:
    """Calculate mean spending per successful order, returning zero for no sales."""
    paid = [order for order in orders if order['status'] == 'paid']
    return total_revenue(paid) / len(paid) if paid else 0.0


def popular_products(order_lines: list[dict], limit: int = 5) -> list[tuple[str, int]]:
    """Rank best-selling products by the number of units sold."""
    quantities = Counter()
    for line in order_lines:
        quantities[line['sku']] += line['quantity']
    return quantities.most_common(limit)
