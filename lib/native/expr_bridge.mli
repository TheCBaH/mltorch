(* The boundary between the expression language and this engine.

   [Expr] owns no tensor, storage or graph type — that is what lets it be an
   independent library — so everything Native-specific about a [Load] lives
   here: which tensor a source symbol names, and how a coordinate is read.
   See .ai/native_expr_refactoring_design.md. *)

val source_of_id : Tensor_id.t -> Expr.Source.t

val id_of_source : Expr.Source.t -> Tensor_id.t
(** A stateless bijection through the integer, not a side table: nothing then
    depends on allocation order, and a printed [Load] renders the same symbol
    either side of the migration. *)

val coord_of_vec6 : 'a Vec6.t -> 'a Expr.Coord.t
val vec6_of_coord : 'a Expr.Coord.t -> 'a Vec6.t

val env : binding:(Tensor_id.t -> Tensor.packed option) -> Expr.Eval.Env.t
(** Supplies [Expr.Eval] with the value at a source and coordinate.

    The binding is OPTION-valued and the coordinate is bounds-checked here,
    before a [Vec6.coord] is built. Both matter: [Tensor.read_at] raises
    [Invalid_argument] above an extent and [Dim.index] raises on a negative, so
    reusing [Tensor.read_at_raw] directly would let exceptions escape through a
    [Core.result] API — and a total binding could only report a missing tensor
    by raising too. *)
