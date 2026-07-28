"""Token pricing, so cost-per-invoice is computed rather than estimated.

The project's rule is that no cost figure reaches the README unless it came out
of `extraction_runs`. This table is the multiplier that turns measured token
counts into that figure.

Prices are USD per million tokens. Promotional prices carry an expiry: a table
that silently keeps applying an introductory rate after it lapses produces
numbers that look precise and are wrong, which is worse than having none.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

MILLION = Decimal("1000000")

# Cache reads bill at roughly a tenth of the input rate; cache writes at about
# 1.25x for the default 5-minute TTL.
CACHE_READ_MULTIPLIER = Decimal("0.1")
CACHE_WRITE_MULTIPLIER = Decimal("1.25")


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    # Introductory pricing, if any, and the day it stops applying.
    promo_input_per_mtok: Decimal | None = None
    promo_output_per_mtok: Decimal | None = None
    promo_until: date | None = None

    def rates(self, on: date) -> tuple[Decimal, Decimal]:
        if (
            self.promo_until is not None
            and self.promo_input_per_mtok is not None
            and self.promo_output_per_mtok is not None
            and on <= self.promo_until
        ):
            return self.promo_input_per_mtok, self.promo_output_per_mtok
        return self.input_per_mtok, self.output_per_mtok


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-8": Price(Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": Price(
        Decimal("3.00"),
        Decimal("15.00"),
        promo_input_per_mtok=Decimal("2.00"),
        promo_output_per_mtok=Decimal("10.00"),
        promo_until=date(2026, 8, 31),
    ),
    "claude-haiku-4-5": Price(Decimal("1.00"), Decimal("5.00")),
}


def estimate_cost(
    model: str,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    on: date | None = None,
) -> Decimal | None:
    """USD for one call. None when the model has no price entry.

    Returning None rather than zero for an unpriced model matters: a locally
    hosted model genuinely costs nothing per token, and conflating "free" with
    "unknown" would let a missing price entry masquerade as a cost saving.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    rate_in, rate_out = price.rates(on or date.today())
    total = (
        Decimal(tokens_in) * rate_in
        + Decimal(tokens_out) * rate_out
        + Decimal(cache_read_tokens) * rate_in * CACHE_READ_MULTIPLIER
        + Decimal(cache_write_tokens) * rate_in * CACHE_WRITE_MULTIPLIER
    ) / MILLION
    return total.quantize(Decimal("0.000001"))
