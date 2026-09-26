(* See node_executor.mli. *)

open Graph_ir

type t = {
  run :
    'e.
    graph ->
    node ->
    output:Output_ordinal.t ->
    out_shape:Vec6.shape ->
    operands:Tensor.packed Tensor_id.Map.t ->
    direct:(unit -> (Tensor.packed, 'e) Err.t) ->
    (Tensor.packed, 'e) Err.t;
}

let default =
  { run = (fun _ _ ~output:_ ~out_shape:_ ~operands:_ ~direct -> direct ()) }
