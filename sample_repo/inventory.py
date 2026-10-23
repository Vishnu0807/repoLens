"""In-memory stock management for the example shop."""
from dataclasses import dataclass


@dataclass
class StockItem:
    sku: str
    available: int
    reserved: int = 0

    def remaining(self) -> int:
        """Count units that are still available for new orders."""
        return self.available - self.reserved


def reserve_stock(item: StockItem, quantity: int) -> StockItem:
    """Reserve inventory before checkout and prevent overselling."""
    if quantity < 1 or quantity > item.remaining():
        raise ValueError('Insufficient stock or invalid quantity')
    return StockItem(item.sku, item.available, item.reserved + quantity)


def release_stock(item: StockItem, quantity: int) -> StockItem:
    """Release reserved inventory after cancellation or failed checkout."""
    if quantity < 1 or quantity > item.reserved:
        raise ValueError('Invalid release quantity')
    return StockItem(item.sku, item.available, item.reserved - quantity)


def low_stock(items: list[StockItem], threshold: int = 5) -> list[str]:
    """List product SKUs that need replenishment below a stock threshold."""
    return [item.sku for item in items if item.remaining() < threshold]
