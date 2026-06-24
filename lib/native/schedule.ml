(* The external iterator: it owns the loop over the output coordinate space and is
   independent of any op. For now a single naive schedule (C innermost) that
   materialises a fresh dense float32 tensor; tiled/parallel/vectorised variants
   plug in here later. See .ai/native_compute_design.md §5.

   A Direct op pixel is [(Axis.t -> int) -> float]; [evaluate] supplies each
   output coord's components as that index function. *)

let for_each = Vec6.iter
let coord_ix (c : Vec6.coord) (a : Axis.t) : int = Dim.to_int (Vec6.get c a)

let evaluate (shape : Vec6.shape) (pixel : (Axis.t -> int) -> float) :
    Tensor.packed =
  Tensor.materialize shape (fun c -> pixel (coord_ix c))

(* The Symbolic counterpart: an op's [pixel] built once at [Symbolic] is an
   [Expr.t] with no data of its own — [ground] is what turns that expression
   back into a concrete tensor, by evaluating it at every output coord against
   a [binding] that supplies real data for each [Tensor_sig.t] it was built
   over. The same expression can be grounded against different bindings
   without rebuilding it. *)
let ground (shape : Vec6.shape) ~(binding : Tensor_sig.t -> Tensor.packed)
    (e : Expr.t) : Tensor.packed =
  Tensor.materialize shape (fun c -> Expr.eval ~binding ~coord:(coord_ix c) e)
