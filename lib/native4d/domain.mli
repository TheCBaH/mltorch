(* Is this Native graph inside the Native4D dialect's domain? The partiality
   contract of .ai/native4d_design.md §1, checkable before any Native4D IR
   exists — it only inspects the Native graph, which is why it lands first (see
   .ai/native4d_plan.md stage 0).

   Takes a VALIDATED VIEW, not a graph. Given a graph it would have to call
   [Graph_view.of_graph] itself and so return [Graph_view.error] too, which
   [Error.t] cannot express — and a `Source_view case would make every caller
   handle a failure it has already ruled out. Both real callers hold a view
   already: the lowerer receives a snapshot, whose [Snapshot.view] is exactly
   this argument, and the tests validate their fixtures once before checking
   them. The view is the framework's trust boundary (graph_view.mli), so
   consuming it is what keeps this module's error row purely about the dialect.

   A NEGATIVE ANSWER IS NOT A BUG. Native4D is a deliberately reduced
   representation; a graph it rejects is outside its domain, not a gap in its
   implementation.

   Meaningful only on a CANONICAL graph. The PT2 importer emits conv weights
   right-aligned from ATen's [out, in, kh, kw], which lands on D/H/W/C — every
   conv weight in an unpipelined resnet18 has non-unit D — so this check would
   reject the model outright. It is the relayout passes plus constant folding
   that produce [N=out, H=kh, W=kw, C=in]. Run [Pipeline.canonical] first. *)

val check : Graph_view.t -> (unit, Error.t) Err.t
