Function: `Strategy.on_tick` (vc_test.py:56) (node:stable:sym:vc_test.py:Strategy.on_tick)
Signature: `on_tick(self, x: float)`

1) Integrated explanation (GUESS)
`Strategy.on_tick` is a function that accepts `self` and a float `x`. No direct callers are identified within the current graph, suggesting it may function as an entry point or be invoked by external components. Internally, it deterministically calls `Strategy.helper` and `util_log`. It also performs two virtual dispatches, `self.meter.update` and `target.update`, which may target either `stable:sym:vc_test.py:AverageMeter.update` or `stable:sym:vc_test.py:ChartItem.update`. During its execution, it interacts with standard library functions like `len` and `print`, with the `print` interaction resulting in an I/O console side effect (**GUESS**).

2) Proven call-graph facts
- **Callers**: none in graph.
- **Callees**: `Strategy.helper` (edge:stable:sym:vc_test.py:Strategy.on_tick->stable:sym:vc_test.py:Strategy.helper), `util_log` (edge:stable:sym:vc_test.py:Strategy.on_tick->stable:sym:vc_test.py:util_log)
- **Virtual dispatch**: may target `stable:sym:vc_test.py:AverageMeter.update` (node:stable:sym:vc_test.py:AverageMeter.update), `stable:sym:vc_test.py:ChartItem.update` (node:stable:sym:vc_test.py:ChartItem.update) based on CALLS_VIRTUAL resolution.
- **Overrides**: none.

3) External interactions & side effects
- **Externals**: local_obj: `self.meter.update` [ns=self] (node:ext:self.meter.update), `target.update` [ns=target] (node:ext:target.update); stdlib: `len` [ns=len] (node:ext:len), `print` [ns=print] (node:ext:print)
- **Side effects (modelled)**: `print` (node:ext:print, side_effect_category=io.console, confidence=0.45)
- **Downstream paths**:
  - `Strategy.on_tick -> Strategy.helper`
  - `Strategy.on_tick -> Strategy.helper -> util_log`
  - `Strategy.on_tick -> util_log`
  - `Strategy.on_tick -> vc:05f9f00142cb[self.meter.update]→[stable:sym:vc_test.py:AverageMeter.update, stable:sym:vc_test.py:ChartItem.update]`
  - `Strategy.on_tick -> vc:6d9007379bb4[target.update]→[stable:sym:vc_test.py:AverageMeter.update, stable:sym:vc_test.py:ChartItem.update]`

4) Code excerpt
```python


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
```