from __future__ import annotations

"""
Test file for VirtualCall vs deterministic CALLS resolution.

What you should see in the graph:

Deterministic:
- Strategy.on_tick -> Strategy.helper
- Strategy.helper -> util_log

Virtual calls:
- Strategy.on_tick -> vc:...  (for self.meter.update)
- Strategy.on_tick -> vc:...  (for target.update)

Potential VC targets:
- AverageMeter.update
- ChartItem.update
"""

# --- some simple helpers / external-ish calls --------------------------------


def util_log(msg: str) -> None:
    print(msg)  # should be a CALLS to External("print")


# --- two classes that both define `update` ------------------------------------


class AverageMeter:
    def update(self, value: float) -> None:
        # pretend to accumulate some metric
        util_log(f"AverageMeter updating with {value}")


class ChartItem:
    def update(self, value: float) -> None:
        # pretend to refresh a chart
        util_log(f"ChartItem updating with {value}")


# --- strategy that mixes deterministic and virtual-style calls ----------------


class Strategy:
    def __init__(self, meter: AverageMeter | ChartItem, chart_item: ChartItem) -> None:
        self.meter = meter          # could be AverageMeter or ChartItem
        self.chart_item = chart_item

    def helper(self, x: float) -> None:
        # This should resolve deterministically:
        # Strategy.helper -> util_log as a normal CALLS edge.
        util_log(f"helper called with {x}")

    def on_tick(self, x: float) -> None:
        # 1) Deterministic internal call: Strategy.on_tick -> Strategy.helper
        self.helper(x)

        # 2) Dynamic-ish dispatch on attribute of self:
        #    self.meter may be AverageMeter or ChartItem.
        #    Your builder should *not* be able to deterministically resolve
        #    'self.meter.update', so this should create a VirtualCall.
        self.meter.update(x)

        # 3) Dynamic-ish dispatch via local variable alias:
        #    `target` is a local alias for self.chart_item.
        #    Again, static resolution of 'target.update' should fail,
        #    so this should produce another VirtualCall.
        target = self.chart_item
        target.update(x)

        # 4) Simple stdlib-ish external: External("len")
        #    classified as stdlib or unresolved depending on your heuristics.
        buffer = [1, 2, 3]
        size = len(buffer)

        util_log(f"tick processed, buffer size = {size}")
