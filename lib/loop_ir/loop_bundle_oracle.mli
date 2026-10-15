(** The Loop interpreter run over a whole {!Loop_bundle.t}, one invocation at a
    time in schedule order. At {!Loop_numerics.Precision.F32} it is the golden
    output of the [Fp32_ordered] policy: every engine that runs the bundle under
    that policy (C, CompCert, Rivet, Wasm) is bitwise equal to it. Each
    invocation runs its own scalar program ([Loop_sum.program]); nothing here
    plans, vectorizes or contracts. *)

open Graph_ir

type error =
  [ `Failed of int * Loop_interp.error
    (** the failing invocation's position in the schedule *)
  | `Missing of Tensor_id.t  (** a constant or input the caller did not bind *)
  ]

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?precision:Loop_numerics.Precision.t ->
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  inputs:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed list, [> error ]) Err.t
(** The graph outputs, in graph output order. [precision] defaults to binary32.
*)
