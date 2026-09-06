# Tail-call backend experiment

This standalone experiment reproduces the recursive structure of expression
evaluation without linking or copying the repository's Expr implementation.
Its independent AST has binary and unary values, selects, and value guards;
the direct evaluator has the same value/guard recursion and the same pending
work after child calls. The executable checks a 200,000-level input, and its
expected outputs make compiler behaviour a test rather than a demonstration.
The Dune stanzas deliberately have no `(libraries ...)` dependency: only the
standalone modules and the OCaml standard library are available to them.

From the repository root:

```sh
make tailcall.runtest
make tailcall.js-benchmark       # compare implementations on both JS backends
```

The cases distinguish:

- self tail recursion;
- statically known mutual tail recursion;
- the proposed `Foo | Bar` rewrite of that mutual pair into one self-tail loop;
- a tag-only refinement that carries common arguments as loop parameters;
- a tail call through a ref (an unknown call);
- direct binary, unary, and mutually recursive value/guard evaluation;
- the proposed single variant dispatcher over those same three trees;
- eager and delayed closure trampolines over those same trees; and
- a defunctionalized-continuation machine over the same complete standalone
  evaluator, in both allocating-list and reusable-array forms.

The source-shape correspondence is:

| Expr evaluator | Standalone evaluator | Pending work after recursion |
|---|---|---|
| `Value.Binary` | `Binary` | apply the binary operator |
| `Value.Unary` / `Value.Round_f32` | `Unary` | apply the unary conversion |
| `Value.Select` + `Bool.Value_lt` | `Select` + `Value_lt` | compare operands, then choose a branch |

These are structural equivalents, not copies of Expr's types or primitive
semantics. That keeps the test isolated while retaining the call-stack shape
whose backend behaviour it is intended to measure.

The js_of_ocaml CPS build is separate from its default build.  This shows that
CPS makes the unknown *tail* call safe, but does not turn the evaluator's
non-tail recursion into a constant-stack computation.

The `Foo | Bar` case is the precise proposed transformation. It succeeds on
Melange where the two-function mutual case exhausts V8's stack. The larger
evaluator dispatcher is a separate negative control: combining function names
does not eliminate child calls that have an operation pending after return.
The tag-only refinement tests whether compatible function arguments can be
carried separately to avoid allocating a payload variant at every transition.
Every intended tail transition in these control cases is annotated with
`[@tailcall]`, making OCaml warning 51 a compile-time check of the premise.

The delayed trampoline counts transitions through evaluation, guard evaluation,
and continuation application. At its threshold it returns a suspension all the
way to the root driver. The driver invokes it only after the previous stack has
unwound, and the next segment starts at depth zero. The benchmark prints both
the number of root bounces and the maximum segment depth.

The reusable-frame evaluator stores frame tags, expressions, operators, and
partial values in parallel arrays. The arrays grow geometrically during warm-up
and are then reused, amortizing frame allocation across repeated evaluations.
The hybrid evaluator retains direct recursion below a cutoff and switches only
the cutoff subtree to those reusable frames. The existing direct prefix stays
on the stack but cannot grow further, so the cutoff plus fixed machine overhead
bounds the host stack without requiring advance knowledge of dynamic depth.

`backend_driver.ml` is also a small cppo proof. Native selects the mutual
functions and both JavaScript compilations select the tag-and-arguments state
loop. This preserves the direct native form while using a form that is safe on
both JavaScript compilers without allocating a payload state per transition.
Each backend has a separate Dune compilation because cppo runs while OCaml
bytecode is built: it cannot discover that the same bytecode library will later
be linked into a `(modes js)` executable.

See [the design record](../../.ai/expr_tailcall_design.md) for the results,
JavaScript profiles, and production refactoring options.
