(* Vocabulary and layout helpers shared by every normalisation in the [Norm]
   category ([norm.ml]): the [Target] a shared diagnostic came from, and the
   NORMALIZED-axes layout/count that [Norm_rms], [Norm_layer] and
   [Norm_group] each reuse rather than restate. *)

(* Which normalisation a shared diagnostic came from.

   Both importers report the same two [normalized_shape] faults for three ATen
   targets, and until Group 7 the message said "rms_norm" unconditionally --
   correct while rms_norm was the only caller and wrong the moment layer_norm
   reused [Op_bridge.normalized_dims]. A closed set, so a variant rather than
   the [string] that [Invalid_dim]/[Dims_count] carry (CLAUDE.md: a closed
   value set is a variant). Here rather than in either importer because both
   need it and two copies is one drift away from two vocabularies. *)
module Target = struct
  type t = Layer_norm | Native_layer_norm | Rms_norm

  let pp fmt = function
    | Layer_norm -> Fmt.string fmt "layer_norm"
    | Native_layer_norm -> Fmt.string fmt "native_layer_norm"
    | Rms_norm -> Fmt.string fmt "rms_norm"
end

(* The layout ATen gives an operand indexed by the NORMALIZED axes only: the
   input's extent on each of [dims], extent 1 (broadcast) everywhere else.
   rms_norm's weight and layer_norm's weight and bias all have it, and one
   definition is what keeps shape inference, the operand check and the walk's
   operand synthesis from disagreeing about it. *)
let normalized_shape ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  List.fold_left
    (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    dims

(* How many elements one normalisation reduces over -- the divisor of the mean
   and of the mean of squares.

   BOUNDED, and that is not decoration. [lib/native] is js_of_ocaml-reachable,
   where [int] is 32 bits, and this is a product of up to six extents each
   bounded only by [Kernel.Limits.Hard.extent] (2^31): the mathematical product
   reaches 2^186, so folding it into an [int] and checking afterwards is a
   post-overflow comparison, not a bound (CLAUDE.md's aggregate rule).
   [Vec6.numel_bounded] divides each factor into the ceiling BEFORE multiplying,
   so no intermediate wraps.

   Called from [output_shape], which is where [Graph_builder] and a JSON-decoded
   graph both meet it -- a rule only the importers enforced would be a rule the
   other two constructors do not have. *)
let normalized_count ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
    (normalized_shape ~x_shape ~dims)

(* The same product as an [int], for [Compute], which has no error channel.

   Sound as a plain fold ONLY because [output_shape] has already run
   [normalized_count] on this node: its result is below [Kernel.Limits.Hard.numel]
   = 2^31, which is exactly [max_int] on the 32-bit backend, so this product
   cannot wrap there. A node whose shape rule never ran cannot be evaluated, so
   the precondition holds by construction -- but it IS a precondition, and this
   function must not be called anywhere the bound has not been established. *)
let normalized_count_unchecked ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 dims
