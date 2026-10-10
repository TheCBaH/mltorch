Group 3 (op3.md): `sub.Tensor`, `_unsafe_view.default` and `transpose.int`
through the payload-free export path.

UNGATED and hand-built, for the same reason as `me_group2_cram.t`: no model
this repository can download serializes any of these three targets (every
release model is exported post-decomposition), so this is the only place they
reach Model Explorer at all.

Three small models, following `unbind_native_cram.t`'s per-case style rather
than `me_group2_cram.t`'s single chained pipeline: Group 3's targets do not
compose into a realistic layer sequence the way conv/pool/norm/linear do.

`group3.json` chains a broadcast `sub.Tensor`, a scalar-spelled `sub.Tensor`,
and a rank-changing `_unsafe_view.default` carrying a `-1` -- every node stays
inside the four-axis dialect, so this is the ACCEPTED case.

`group3-transpose.json` and `group3-transpose-refused.json` isolate
`transpose.int`'s two questions separately, per op3-impl.md's round-1 review
response: both use `[1,h,w,c]` with DISTINCT, non-unit `h`/`w`/`c` (a
conventional `[2,3,4,5]` would put a real batch on `D` and answer the
shape-domain question instead of the axis-domain one). `dims=(1,2)` stays
inside the dialect; `dims=(0,1)` names axis `D` and is refused by the axis
check specifically, not by the shape check.

  $ ./cram_probe.exe fixture group3

Each model's outcome, and the full capability vector for `group3.json`: every
stage available, Native4D included -- the accept case op3-impl.md's exit
condition asks for. `group3-malformed` confirms, rather than assumes,
op3-impl.md's claim that Me_classify needs no Group-3 change: it matches
`#Native_interp.malformed` wholesale, so commit 1's new `` `Bad_view `` row is
classified already, and reads exactly like every other malformed row here --
refused before a session is built at all, which is what a decoder that
accepted a graph it should not have is supposed to do.

  $ for m in group3 group3-transpose group3-transpose-refused group3-malformed; do
  >   printf '%-28s ' "$m"
  >   if ../bin/native_graph.exe visualize --model $m.json --output $m.session.json >/dev/null 2>err.txt; then
  >     echo lowered
  >   else
  >     head -1 err.txt
  >   fi
  > done
  group3                       lowered
  group3-transpose             lowered
  group3-transpose-refused     lowered
  group3-malformed             native_graph: malformed PT2 graph: view size [-1, -1]: view size [-1, -1] has more than one inferred (-1) dimension

  $ caps() {
  >   ./cram_probe.exe brief "$1"
  > }
  $ caps group3.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps group3-transpose.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps group3-transpose-refused.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis D is outside the N/H/W/C dialect

The IMPORTED native graph for `group3.json`, with every op's parameters: the
broadcast operand's own shape, the scalar sub's negated value, and the
`_unsafe_view` target -- an importer that dropped the broadcast, used `+3`
instead of `-3`, or resolved the `-1` from the wrong product would show up
here.

  $ ./cram_probe.exe params 8 group3.session.json
  n0  Sub      sub a=t0 b=t1
  n1  Add_scalar add_scalar x=t2 scalar=-3
  n2  Reshape  reshape x=t3 params={shape=[W=8 C=32]}

Output metadata carries the shape a reader checks each parameter against --
the broadcast sub's output stays at `x`'s shape (leading `D=1` trimmed by
`Vec6.pp_shape`, not carried away), the scalar sub leaves it unchanged, and
the `_unsafe_view` target is exactly what `[-1, 32]` resolves to against 256
elements.

  $ ./cram_probe.exe shapes 10 group3.session.json
  n0  Sub        [H=4 W=8 C=8]
  n1  Add_scalar [H=4 W=8 C=8]
  n2  Reshape    [W=8 C=32]

Stable slot ids and wiring order: `other` is a captured parameter (the
broadcast weight a real model would serialize this way), so its edge sources
from a CONSTANT input, not the graph's user input.

  $ ./cram_probe.exe edges 10 group3.session.json
  n0  Sub        [('in:t0', '0', 't0'), ('const:t1', '0', 't1')]
  n1  Add_scalar [('n0', '0', 't2')]
  n2  Reshape    [('n1', '0', 't3')]

The `transpose.int` permutation itself, for the accepted case: `dims=(1,2)`
swaps `H` and `W`.

  $ ./cram_probe.exe params 8 group3-transpose.session.json
  n0  Permute  permute x=t0 perm=[H<-W, W<-H]
