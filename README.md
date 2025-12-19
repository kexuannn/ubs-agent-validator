# AlgoTracer

AlgoTracer builds a Python call graph inside Memgraph and explains what a function does using only graph facts — not guesses, not static summaries.

Unlike static analyzers that over-resolve everything, AlgoTracer treats uncertain instance/class calls as virtual calls and resolves them safely when needed.

## 🧠 Problem & Motivation

Python codebases are full of calls like:

- `self.process()`
- `engine.send_order()`
- `cls.build()`

Without full type inference, resolving these statically is:
- ❌ Wrong — you might point to the wrong function
- ❌ Expensive — the graph explodes with edges
- ❌ Unhelpful — too many “maybe” relationships

## ✔️ AlgoTracer’s Solution
- **During ingest** → record `VirtualCall` instead of guessing
- **During explain** → use actual class hierarchy + overrides
- **Result** → accurate, bounded possible callees

## ✅ What AlgoTracer Does

### 1️⃣ Parse Your Python Repo
- Scans `.py` and `.ipynb`
- Extracts modules, classes, functions/methods, overrides, calls (deterministic + virtual), decorators

### 2️⃣ Build a Memgraph-First Code Graph
Your repo becomes a compact, queryable graph.

**Node Types**

| Node        | Meaning                        |
|-------------|--------------------------------|
| Repo        | Logical repository             |
| Module      | Python file                    |
| Class       | Class declaration              |
| Function    | Function or method             |
| External    | Unresolved / outside-repo call |
| VirtualCall | Deferred instance/class call   |

**Edge Types**

| Edge          | Meaning                           |
|---------------|-----------------------------------|
| HAS_MODULE    | Repo → Module                     |
| DEFINES       | Module/Class → Function           |
| DEFINES_CLASS | Module → Class                    |
| SUBCLASS_OF   | Class inheritance                 |
| OVERRIDES     | Child method → Parent method      |
| CALLS         | Deterministic call                |
| CALLS_VIRTUAL | Uncertain / polymorphic call      |

### 🧩 Example — Code → Graph

**Code**
```python
class Base:
    def run(self):
        self.execute()

class Child(Base):
    def execute(self):
        print("hi")
```

**Stored Graph Relationships**
- `Class(Base) -[:DEFINES]-> Function(Base.run)`
- `Class(Child) -[:SUBCLASS_OF]-> Class(Base)`
- `Function(Base.run) -[:CALLS_VIRTUAL]-> VirtualCall(execute)`
- `Function(Child.execute) -[:OVERRIDES]-> Function(Base.execute?)` (best-effort)

During explanation:
- Detects `self.execute()`
- Sees caller belongs to `Base`
- Walks inheritance + overrides
- Finds `Child.execute()`
- Returns bounded “possible callees”

## 🚀 Quickstart

1️⃣ **Start Memgraph**
```bash
docker run -it --rm \
  -p 7687:7687 -p 3000:3000 \
  memgraph/memgraph-platform
```
- Bolt: `127.0.0.1:7687`
- UI (optional): `http://localhost:3000`

2️⃣ **Install AlgoTracer**
```bash
pip install -e .
```

3️⃣ **Build the Graph**
```bash
algotracer analyze path/to/repo --repo-id my-repo
```
Notes:
- Skips `tests/` unless `--include-tests`
- Supports `.ipynb` via safe conversion

4️⃣ **Explain a Function**
```bash
algotracer explain path/to/repo \
  --name Class.fit \
  --file src/model.py \
  --repo-id my-repo \
  --output reports/explanation.md
```
- Uses Gemini if available (GEMINI_API_KEY + google-generativeai)
- Otherwise deterministic summary
- Always grounded in graph truth

## 🧠 How Virtual Calls Work

| Stage   | What happens                                  |
|---------|-----------------------------------------------|
| ingest  | detect `self.foo()` / `cls.bar()` / etc       |
| graph   | store `CALLS_VIRTUAL` → `VirtualCall`         |
| explain | resolve possible receivers using class graph  |
| output  | produce bounded “may call” targets            |

Included heuristics:
- `self.method()`
- `cls.method()`
- `super().method()`
- Named receiver fallback where possible

## 🗺️ Graph Navigation Cheatsheet

| Concept              | Graph Pattern                             |
|----------------------|-------------------------------------------|
| Repo structure       | `Repo -[:HAS_MODULE]-> Module`            |
| Functions in module  | `Module -[:DEFINES]-> Function`           |
| Class methods        | `Class -[:DEFINES]-> Function`            |
| Inheritance          | `Class -[:SUBCLASS_OF]-> Class`           |
| Overrides            | `Function -[:OVERRIDES]-> Function`       |
| Deterministic call   | `Function -[:CALLS]-> Function/External`  |
| Virtual call         | `Function -[:CALLS_VIRTUAL]-> VirtualCall`|

## 🧪 CLI Reference

**Analyze**
```
algotracer analyze <path>
  --repo-id ID
  --include-tests
  --memgraph-host
  --memgraph-port
  --memgraph-user
  --memgraph-password
```

**Explain**
```
algotracer explain <path>
  --id OR (--name + --file)
  --depth-up
  --depth-down
  --max-nodes
  --max-edges
  --max-paths
  --output
  --no-llm
```

## 📝 Notes
- Multiple repos can coexist (unique `repo-id`)
- Requires Memgraph running
- Notebook parsing is sandbox-safe
- LLM is optional

## How virtual calls are resolved (during explain)
- Ingest stores uncertain calls as `VirtualCall` nodes instead of guessing targets.
- `CALLS_VIRTUAL` edges originate from functions to these virtual call sites.
- During `explain`, the neighborhood fetcher:
  1) Collects `VirtualCall` nodes in the bounded subgraph.
  2) Seeds possible receiver types:
     - `self/cls/super`: uses the caller’s `owner_class` and its bases (`SUBCLASS_OF`).
     - `name`: tries matching a class name in the repo (best-effort).
  3) Looks up methods named `callee_attr` on those types (and bases) using `Class->Function` and `OVERRIDES`.
  4) Returns `virtual_targets` (virtual call id + possible function ids), bounded and repo-scoped to avoid blowups.
- Deterministic `CALLS` edges are used as-is; virtuals are expanded on demand for explanation.

## How to read the graph (at a glance)
- Modules: `Repo-[:HAS_MODULE]->Module (stable:mod:<path>)`.
- Classes: `Module-[:DEFINES_CLASS]->Class`, `Class-[:SUBCLASS_OF]->Class`, `Class-[:DEFINES]->Function` (methods).
- Functions: `Function {id=stable:sym:<path>:<qualname>, owner_class?, kind, lineno, signature}`; `Module-[:DEFINES]->Function`, `Class-[:DEFINES]->Function` for methods.
- Overrides: `Function-[:OVERRIDES]->Function` (child -> parent method; best-effort).
- Calls (deterministic): `Function-[:CALLS]->Function|External` when resolvable; side-effects also use External nodes.
- Virtual calls: `Function-[:CALLS_VIRTUAL]->VirtualCall` when we deliberately defer resolution.

## Connection settings
- Env vars: `MEMGRAPH_HOST` (default `127.0.0.1`), `MEMGRAPH_PORT` (default `7687`), `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD` (optional)
- CLI flags override env: `--memgraph-host/--memgraph-port/--memgraph-user/--memgraph-password`.

