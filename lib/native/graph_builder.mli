(* A functional (state-monad) builder for the native graph IR. The state — fresh
   id counters, the accumulated nodes/edges/inputs, and the default element type —
   is threaded purely; nothing is mutated. Output shapes are COMPUTED (via
   [Graph_shape]), never supplied. See .ai/native_graph_design.md. *)

open Graph_ir

type error =
  [ Graph_shape.error | `Expected_single_output_shape of output_count ]

and output_count = { count : int }

val pp_error : Format.formatter -> [< error ] -> unit

type 'a t

val return : 'a -> 'a t
val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t

(* An external input edge. [?name] is retained as an ignored compatibility
   argument; native graph identity is always the allocated [Tensor_id]. [?fmt]
   defaults to the builder's default element type (F32 unless [build] overrides
   it). *)
val input :
  shape:Vec6.shape ->
  ?name:string ->
  ?fmt:Payload.packed_fmt ->
  ?quant:Quant.t ->
  unit ->
  Tensor_id.t t

(* A captured, read-only inference tensor. Its model/archive payload association
   belongs to the importer sidecar, not to the generic native graph. *)
val constant :
  shape:Vec6.shape ->
  ?name:string ->
  ?fmt:Payload.packed_fmt ->
  ?quant:Quant.t ->
  unit ->
  Tensor_id.t t

(* Typed-operand op constructors: each appends a node and returns its fresh output
   edge. [?name] is an ignored compatibility argument. Omitting an optional
   operand ([?bias], [?weight]) records [None] in the IR; the evaluator fills the
   identity (zeros bias / ones weight). *)
(* Op constructors in global alphabetical order (see graph_ir.mli). *)
val add : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

val avg_pool2d :
  ?name:string -> Pool.AvgPool2d.params -> tensor_ref -> Tensor_id.t t

val batch_norm :
  ?name:string ->
  Norm.BatchNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  ?bias:tensor_ref ->
  running_mean:tensor_ref ->
  running_var:tensor_ref ->
  unit ->
  Tensor_id.t t

val bmm : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

val conv2d :
  ?name:string ->
  Conv.Conv2d.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val conv2d_padding :
  ?name:string ->
  Conv.Conv2d_padding.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val convolution :
  ?name:string ->
  Conv.Convolution.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val div : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

(* Route a dead edge into a [Discard] sink node (no output). Used to keep a
   multi-output op's full arity while marking an unused result for later pruning. *)
val discard : tensor_ref -> unit t

val linear :
  ?name:string ->
  Linear.Linear.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val max_pool2d :
  ?name:string -> Pool.MaxPool2d.params -> tensor_ref -> Tensor_id.t t

(* max_pool2d_with_indices returns two edges: (values, indices). *)
val max_pool2d_with_indices :
  ?name:string ->
  Pool.MaxPool2dWithIndices.params ->
  tensor_ref ->
  (Tensor_id.t * Tensor_id.t) t

val mean : ?name:string -> Reduce.Mean.params -> tensor_ref -> Tensor_id.t t
val mul : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

val permute :
  ?name:string -> Permute.Permute.perm -> tensor_ref -> Tensor_id.t t

val relu : ?name:string -> tensor_ref -> Tensor_id.t t

val reshape :
  ?name:string -> Reshape.Reshape.params -> tensor_ref -> Tensor_id.t t

val rms_norm :
  ?name:string ->
  Norm.RmsNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  unit ->
  Tensor_id.t t

val sqrt : ?name:string -> tensor_ref -> Tensor_id.t t
val sub : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

(* Structurally group nodes emitted by [body].  The group shares the enclosing
   graph's global SSA namespace and has no inputs, outputs, or call semantics. *)
val group : ?label:string -> 'a t -> 'a t

(* Run a builder computation from the empty state and finalise into a graph;
   [outputs] selects the graph outputs from the computation's result. [?dtype] is
   the default element type for [input] (default F32). [name] is ignored. *)
val build :
  ?dtype:Payload.packed_fmt ->
  name:string ->
  outputs:('a -> Tensor_id.t list) ->
  'a t ->
  (graph, error) Core.result
