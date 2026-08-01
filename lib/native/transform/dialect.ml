(* What a generic graph algorithm needs to know about a dialect.

   Small on purpose. .ai/native4d_design.md §5.2 asks for "a small, explicit
   dialect interface", and the measure of success is how little is in it: the
   structural records are shared (see [Graph_common]), the symbolic term
   language is dialect-free, and the cluster algebra is keyed on ids. What is
   left is the handful of questions only the dialect can answer about its own
   operations.

   Kept CAPABILITY-BASED, per §5.2: a printer or a relation algebra should not
   have to depend on evaluation. [symbolic] is here because [Map_verify] needs
   it, and a dialect that only ever wanted validation could satisfy a narrower
   signature — which is why [Side.S] bundles the evaluation-facing parts
   separately rather than growing this one. *)

module type S = sig
  type op

  (* The dialect's own shape-error row. Abstract here, which is why every
     consumer WRAPS it rather than inheriting it: OCaml can inherit only a type
     known to expand to a polymorphic-variant row. See [Graph_view.error]. *)
  type shape_error

  val pp_shape_error : Format.formatter -> shape_error -> unit

  (* A generic validator has to look signatures up and can hit a missing one,
     but it cannot BUILD a value of an abstract row — so the dialect supplies
     the constructor. Every dialect's shape row already has this case, since
     shape inference needs it too. *)
  val missing_sig : Tensor_id.t -> shape_error
  val operands : op -> Tensor_id.t list
  val map_operands : (Tensor_id.t -> Tensor_id.t) -> op -> op

  (* In the six-axis frame, whatever the dialect's own shape type: the generic
     validator only compares against the stored [Tensor_sig.shape], so a
     four-axis dialect converts on the way out and [Shape4] never leaks into
     this interface. *)
  val output_shape :
    op ->
    sig_of:(Tensor_id.t -> (Tensor_sig.t, shape_error) Core.result) ->
    (Vec6.shape list, shape_error) Core.result

  (* The dialect's standing invariant on ANY signature, node-produced or not.
     Native answers [Ok]; Native4D rejects extent on T or D.

     Without this hook the four-axis invariant leaks. [Graph_view] constrains
     shapes only where an OP produces them — it compares each node's outputs
     against [output_shape] — and a graph input or a captured constant is
     produced by no op, so a graph whose input is directly its output would
     validate with a signature the dialect forbids. *)
  val validate_sig : Tensor_sig.t -> (unit, shape_error) Core.result
  val classify : op -> output:int -> Output_transfer.t
  val pp_op : Tensor_id.t Fmt.t -> Format.formatter -> op -> unit
end
