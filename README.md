
# AlgoTracer — Static Python Call Graph & Virtual Dispatch Analyzer

AlgoTracer is a **static analysis engine** that parses Python source code, builds a structured **call graph**, identifies **static calls**, **virtual / polymorphic callsites**, and explains code behavior using a deterministic evidence-driven explainer with optional LLM reasoning layers.

This is inspired by production‑grade program analyzers used in security scanning, compilers, financial trading backtest engines, and IDE intelligence systems — but purpose‑built for real‑world messy Python repos (frameworks, strategies, inheritance, duck‑typing, dynamic assignment).

---

## 1. Core Goals

AlgoTracer is built to answer these questions **with evidence, not guesses**:

- What calls what?
- Where can control flow dynamically branch due to polymorphism?
- Which calls are guaranteed vs. ambiguous?
- Does a call come from inheritance, local dispatch, or attribute lookup?
- What code paths can theoretically trigger execution?

And most importantly:

> “If I look at a function in isolation, what is the full *truth* of how it interacts with the system?”

AlgoTracer produces a **graph-backed truth layer** before ANY AI explanation happens.

The LLM never invents — it only explains what evidence proves.

---

## 2. Architecture Overview

AlgoTracer is composed of 5 main systems:

### 2.1 AST Parser
- Walks Python files and extracts:
  - modules
  - classes
  - functions
  - method bodies
  - imports
  - attribute resolution context
  - call sites

### 2.2 Graph Builder
- Converts AST insights into stable node+edge structures
- Resolves deterministic calls
- Emits VirtualCall nodes for uncertain dispatch
- Handles inheritance and symbol resolution

### 2.3 Memgraph Storage
Graph persisted in Memgraph / Neo4j-style schema:

| Node Type | Meaning |
|--------|--------|
| Module | Python file |
| Class | Class definition |
| Function | Top‑level or method |
| External | Built‑ins / libs |
| VirtualCall | Ambiguous callsite |

| Edge | Meaning |
|------|--------|
| CALLS | Proven direct call |
| CALLS_VIRTUAL | Dynamic dispatch candidate |
| OVERRIDES | Child overrides base method |
| IMPORTS | module import |

### 2.4 Neighborhood Extractor
For any function:
- resolves upstream callers
- downstream callees
- external dependencies
- possible virtual dispatch targets
- supporting context nodes
- source snippet

### 2.5 Explainer Engine
Builds a structured evidence pack.

LLM can be optionally layered to generate narrative explanation while respecting rules:
- No invented calls
- No hallucinated code
- No fake structure
- MUST cite graph evidence

---

## 3. Static Call Resolution Model

### 3.1 Deterministic Calls
A call becomes deterministic when its target can be proven statically.

Examples:

```
self.helper()
foo.bar()
module.function()
MyClass.method()
```

AlgoTracer checks:

- lexical scope → local functions
- class resolution → method table
- inheritance relationships
- imported references
- explicit symbols
- known builtin API

Result → A `CALLS` edge is emitted.

---

## 4. Virtual Calls (Polymorphism + Duck Typing)

When a call target **cannot** be proven, AlgoTracer emits:

```
CALLS_VIRTUAL
```

and creates a `VirtualCall` node with metadata:

- receiver root kind (self / cls / param / local)
- full expression text
- call attribute
- heuristic receiver identity context
- hash‑stable vc id

Examples that become VC:

```
self.meter.update()
target.update()
obj.process()
handler(x)
plugin.run()
strategy.execute()
```

Even better — AlgoTracer attempts to find **possible target candidates**:

- sibling classes with same method
- inherited classes
- polymorphic interface classes
- same‑attribute assignment tracking

If found → VC is annotated:

```
vc:abc123 → [AverageMeter.update, ChartItem.update]
```

If not → unresolved VC but still tracked.

---

## 5. Virtual Call Naming

Instead of mysterious vc hashes, AlgoTracer names them:

```
vc:a3a4db0bf2b3 [self.pos_data.items]
vc:179d10f9269c [set().difference]
```

Sources include:
- full expression text
- receiver attribute pair
- inferred root source

This dramatically improves debuggability.

---

## 6. Explainer System

The explainer constructs a deterministic structured report including:

1️⃣ Function identity  
2️⃣ What calls it  
3️⃣ What it calls  
4️⃣ Virtual dispatch explanation  
5️⃣ Inheritance notes  
6️⃣ Side effects detection  
7️⃣ Upstream + downstream paths  
8️⃣ Source excerpt  

LLM is optional.  
When enabled:
- rewrites only Section 1 narrative
- cannot change facts
- cannot invent nodes
- cannot fabricate calls

---

## 7. Stable Paths

Every function is stored as:

```
stable:sym:<path>:<qualname>
```

Example:

```
stable:sym:tests/vc_test.py:Strategy.on_tick
stable:sym:vnpy/alpha/strategy/strategies/equity_demo_strategy.py:EquityDemoStrategy.on_bars
```

This guarantees graph identity even across rebuilds.

---

## 8. Source Snippets

AlgoTracer extracts real source from disk when explaining.

- Finds owning module
- Uses abs path or repo root
- Captures context lines
- Detects empty snippet cases
- Prints debug summary
- Injects snippet into evidence pack

This enables **evidence‑anchored intent explanation**, not hallucinated descriptions.

---

## 9. Why This is Unique

Unlike normal call graph tools which either:
❌ Give up on Python’s dynamic nature  
❌ Pretend uncertain calls are deterministic  
❌ Produce useless spaghetti graphs  
❌ Or hallucinate AI interpretations  

AlgoTracer builds:

- A truth graph
- Deterministic vs uncertain separation
- Explicitly modeled polymorphism
- Human readable insight
- LLM narrative only after verification

This is one of the first practical **evidence‑grounded polymorphic Python analyzers** built for real systems.

---

## 10. Conclusion

AlgoTracer makes Python call behavior:

- **Visible**
- **Structured**
- **Provable**
- **Explainable**

It is already capable of analyzing large real‑world systems such as algorithmic trading strategies, framework internals, and highly polymorphic design architectures.

