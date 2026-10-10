# ATen diff — admitted Native slice

`torch.ops.aten.diff.default` has one `Graph_ir.Diff` node. The curated C++
binding and generated interpreter dispatch retain the upstream schema:
`diff(self, n=1, dim=-1, prepend=None, append=None)`. Absent and explicitly
null optional tensors both decode to None.

Native admits float32 and int64, order zero or one, any axis of a non-scalar
rank up to six, same-rank optional tensors whose other extents match, and
nonempty results. Order zero copies self, ignoring optional values while
still validating their shapes. Order one virtually concatenates prepend,
self and append, then subtracts each element from its successor. Same-dtype
integer subtraction wraps in int64, including past 2^53; integer payloads
never convert through floats. Mixed dtypes at order one, Bool, higher orders
and empty results are explicit typed refusals. Higher float orders need
intermediate binary32 materialization and cannot use an unrounded recursive
subtraction expression.

`Diff.output_shape` owns format, parameter, shape and bounded aggregate checks
for hand-built graphs, decoded graphs and both importers. Sum extents in int64
before narrowing, using the existing concatenation ceiling. Importers check
declared rank before six-axis right-alignment erases leading singleton axes.
Serialized metadata outside FLOAT/LONG/BOOL is refused before its generic
source-format mapping could silently narrow it; Bool reaches the shared
unsupported-format check.

The generic pixel reader clamps every segment load because symbolic selects
evaluate both branches. Float and int64 readers share coordinate logic and
retain separate value domains. Direct exposes its concrete boolean predicate
type so typed integer selection can consume the same index predicate. Graph
construction retains self's format; direct and symbolic execution validate
the shared rule. Symbolic integers produce Stage_i64 and Kernel integer
expressions. Native4D conversion explicitly refuses Diff pending a dialect
counterpart; this does not prevent the independent Native-to-Kernel branch.

Tests compare against the real C++ ATen oracle, with exact integer comparison,
and cover defaults, optional operands, axes, integer wrapping and IEEE values.
Pure tests exercise direct/symbolic execution, graph codec round trips and
typed refusals. Serialized importer tests cover rank loss and malformed
arguments. A generated ATen walk reuses the rank-four shape/dimension recipe,
bounding every extent at two or more so order-one results remain nonempty;
this covers both axis spellings without adding Diff to the metadata backlog.
All consumer checks avoid Python torch.
