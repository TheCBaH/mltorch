(* The external iterator: it owns the loop over the output coordinate space and is
   independent of any op. For now a single naive schedule (C innermost) that
   materialises a fresh dense float32 tensor; tiled/parallel/vectorised variants
   plug in here later. See .ai/native_compute_design.md §5.

   A Direct op pixel is [Vec6.coord -> float]; [evaluate] is a thin, named
   seam over [Tensor.materialize] (which already takes exactly that) so this
   module stays the one place a future schedule variant plugs in. *)

let for_each = Vec6.iter

(* Direct field projection, not [Vec6.get]: called once per axis for every
   output coord (802,816+ times per real conv), and profiling found
   [Vec6.get]'s dispatch a real cost at this call volume — see
   .ai/pt2_inference_perf.md. Returns the field as-is ([Vec6.coord]'s
   components are already [Dim.index Dim.t], validated at construction), not
   [Dim.to_int]: [evaluate] below feeds [Direct]'s [load], whose [idx] wants
   [position index = Dim.index Dim.t] with no re-validation needed. *)
let coord_index_dim (c : Vec6.coord) (a : Axis.t) : Dim.index Dim.t =
  match a with N -> c.n | T -> c.t | D -> c.d | H -> c.h | W -> c.w | C -> c.c

let evaluate (shape : Vec6.shape) (pixel : Vec6.coord -> float) =
  Tensor.materialize shape pixel

(* The Symbolic counterpart: an op's [pixel] built once at [Symbolic] is an
   [Expr.Value.t] with no data of its own — [ground] is what turns that
   expression back into a concrete tensor, by evaluating it at every output
   coord against a [binding] that supplies real data for each edge it was built
   over. The same expression can be grounded against different bindings without
   rebuilding it.

   [binding] is keyed by [Tensor_id.t] and OPTION-valued, and both matter for
   what comes next: [Expr.Eval] reports a missing source as a structured
   [`Unknown_source], while a total binding can only signal one by raising —
   [Tensor_id.Map.find] does exactly that today, before any error path could
   see it. Behaviour-preserving for now, since the old evaluator still wants a
   total signature-keyed function: [None] becomes a named failure rather than a
   [Not_found] escaping from inside a map lookup. *)
let ground (shape : Vec6.shape) ~(binding : Tensor_id.t -> Tensor.packed option)
    (e : Expr.Value.t) =
  let env = Expr_bridge.env ~binding in
  (* [Core.or_raise] once per PIXEL, not per node, and only here.
     [Tensor.materialize] takes [Vec6.coord -> float] and is the engine's
     iteration seam; threading a result through it would make [Direct] -- the
     actual hot path -- pay for the symbolic one. [Direct]'s pixels already
     raise [Invalid_argument] out of [Dim.index], so this matches [ground]'s
     existing contract rather than widening it. The named helper is what keeps
     the deliberate wrapper-drop distinguishable from the defect. *)
  Tensor.materialize shape (fun c ->
      Core.or_raise Expr.Eval.pp_error
        (Expr.Eval.value env
           ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
           e))
