(** The oracle of the vector layer: a vector program expanded into the scalar
    program its semantics define, lane by lane, with no backend involved.

    A vector loop of [lanes] lanes over [\[lo, hi)] becomes
    [q = (hi - lo) / lanes] iterations of a scalar loop, each executing the
    vector body one statement at a time across lanes 0 to [lanes - 1]
    (statement-major, the order a vector instruction sequence has), then the
    scalar remainder. The expansion is built from the vector body alone, not
    from the loop's stored scalar original, so running it against the original
    checks that the vectorization (its dependence and aliasing analysis
    included) preserved values, failures and mark counts. *)

val lane_expr :
  var:Loop_var.t ->
  base:Loop_index.t ->
  temp:(Loop_vector.Temp.t -> int -> Loop_temp.t) ->
  Loop_vector.t ->
  int ->
  float Loop_expr.t
(** Lane [k] of a vector expression at [var = base], as the scalar expression
    the scalar program evaluates there. [temp] names the scalar holding a vector
    temporary's lane. *)

val expand : Loop_vector.program -> Loop_program.t
(** [Invalid_argument] on a vector loop without constant bounds, which the
    verifier rejects first. *)
