(* Reuse an existing alternate-layout edge to make a mixed elementwise operand
   set uniform, instead of speculatively inserting a new relayout. The match
   record, the operand resolution and the rebuild are private. *)

module Match : sig
  type t
  (** One anchor: an elementwise op whose mixed operand layouts a relayout
      already in the graph can make uniform. *)
end

val pattern : Graph_ir.Tensor_id.t -> Match.t Pattern.t
(** Anchored on an elementwise op's output; exposed so a test can ask what the
    pass matches without running the rewrite. *)

val pass : Pass.t
