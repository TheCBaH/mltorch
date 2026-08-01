(* A tensor shape that has been shown to lie in the four-axis dialect.

   ABSTRACT, which is the point. .ai/native4d_design.md §4.2: "Reusing
   [Vec6.shape] directly in every public Native4D constructor would make an
   invalid Native4D graph constructible." Every op payload and every
   shape-inference result carries this instead, so a non-four-axis shape cannot
   reach the IR through the public API at all.

   What it does NOT replace is [Tensor_sig.t]. The stored signature keeps its
   [Vec6.shape], so [Stage_program], [Expr], [Symbolic], [Ground_expr] and the
   whole symbolic stack are reused verbatim — parameterising the tensor
   signature by shape type would functorise all of them for no gain, since §4.1
   says the physical shape is IDENTICAL across the dialect boundary. See
   .ai/native4d_plan.md, correction C3.

   Its error row is its OWN, and callers widen (the idiom [Graph_shape.widen]
   uses). Returning the aggregate [Error.t] would make [Error] depend on
   [Shape4] and [Shape4] depend on [Error]. *)

type t
type error = [ `Non_four_dimensional_shape of Vec6.shape ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Extents are [Dim.extent], so each is already known positive — [Dim.extent]
   raises below 1, there being no empty tensors. Nothing here can fail. *)
val make :
  n:Dim.extent Dim.t ->
  h:Dim.extent Dim.t ->
  w:Dim.extent Dim.t ->
  c:Dim.extent Dim.t ->
  t

val of_ints : n:int -> h:int -> w:int -> c:int -> t

(* The check, and the only way in from a Native shape. *)
val of_vec6 : Vec6.shape -> (t, [> error ]) Core.result

(* The way out, for calling shared compute. Total: a [t] is a [Vec6.shape] whose
   T and D happen to be 1. *)
val to_vec6 : t -> Vec6.shape
val get : t -> Axis4.t -> Dim.extent Dim.t
val set : t -> Axis4.t -> Dim.extent Dim.t -> t
val numel : t -> Dim.count Dim.t
val equal : t -> t -> bool

(* Prints as [N=… H=… W=… C=…] — deliberately unlike [Vec6.pp_shape], which
   trims leading 1s and would render a four-axis shape and a six-axis one
   identically. A test reading the output can tell which type it has. *)
val pp : Format.formatter -> t -> unit
val jsont : t Jsont.t
