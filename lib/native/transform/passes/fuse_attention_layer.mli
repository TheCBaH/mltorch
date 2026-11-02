(* The inverse of [Attention_layer_decompose]: an encoder attention layer spelled
   out in primitive ops becomes the one [Attention_layer] node it stands for.
   The matcher and builder stay private; the module's contract is the pass. *)

val pass : Pass.t
(** Off by default: no pipeline runs it; a caller that grants the layer does.
    The replacement is bit-for-bit identical, so the output's claim is
    [Identical]. *)
