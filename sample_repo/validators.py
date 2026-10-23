"""Input checks for a tiny example shop."""


def valid_email(address: str) -> bool:
    """Return whether an address has a simple email shape."""
    return "@" in address and "." in address.split("@")[-1]


def check_quantity(quantity: int) -> None:
    """Reject non-positive order quantities."""
    if quantity < 1:
        raise ValueError("Quantity must be positive")

