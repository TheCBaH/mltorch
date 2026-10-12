module Dialect = struct
  type t = Native | Native4d

  (* Alphabetical, as every closed vocabulary here. *)
  let all = [ Native; Native4d ]
  let name = function Native -> "native" | Native4d -> "native4d"
  let of_name s = List.find_opt (fun d -> String.equal (name d) s) all
end

module Native4d_failure = struct
  type t =
    | Bundle of Native4d.Eval_direct4.error Loop_ir.Loop_bundle.generic_error
    | Constants of Const_ssa_materialize.error
    | Convert of Native4d.Error.t
    | Reference of Native4d.Eval_direct4.error
    | Snapshot of Graph_view.error

  let pp ppf = function
    | Bundle e -> Loop_native4d.pp_error ppf e
    | Constants e -> Const_ssa_materialize.pp_error ppf e
    | Convert e ->
        Fmt.pf ppf "outside the Native4D domain: %a" Native4d.Error.pp e
    | Reference e -> Native4d.Eval_direct4.pp_error ppf e
    | Snapshot e -> Graph_view.pp_error ppf e
end

type error =
  [ Native_interp.error
  | Loop_ir.Loop_bundle.error
  | `Native4d of Native4d_failure.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Native4d f -> Native4d_failure.pp ppf f
  | #Loop_ir.Loop_bundle.error as e -> Loop_ir.Loop_bundle.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

type t = {
  bundle : Loop_ir.Loop_bundle.t;
  native_graph : Graph_ir.graph option;
  constants : Tensor.packed Tensor_id.Map.t;
  reference :
    (Tensor_id.t * Tensor.packed) list ->
    (Tensor.packed Tensor_id.Map.t, error) Err.t;
}

let native ~config archive =
  let open Err.Syntax in
  let map e = (e :> error) in
  let* lowered = Native_interp.lower_archive archive |> Err.map_error map in
  let g = lowered.Pt2_native_graph.graph in
  let* constants = Native_interp.preload archive lowered |> Err.map_error map in
  let* bundle = Loop_ir.Loop_bundle.build ~config g |> Err.map_error map in
  Err.return
    {
      bundle;
      native_graph = Some g;
      constants;
      reference =
        (fun inputs ->
          Eval_direct.run g
            ~constants:(Tensor_id.Map.bindings constants)
            ~inputs
          |> Err.map_error (fun e -> map (`Eval e)));
    }

let native4d ~config archive =
  let open Err.Syntax in
  let map e = (e :> error) in
  let fail f = Err.map_error (fun e -> `Native4d (f e)) in
  let* lowered = Native_interp.lower_archive archive |> Err.map_error map in
  let* constants0 =
    Native_interp.preload archive lowered |> Err.map_error map
  in
  let* (Native_interp.Transformed t) =
    Native_interp.transform_lowered ~constants:constants0 lowered
      ~passes:[ Pipeline.canonical ~fold:true ]
    |> Err.map_error map
  in
  let* (Snapshot.Pack src) =
    Snapshot.create t.graph |> fail (fun e -> Native4d_failure.Snapshot e)
  in
  let* (Native4d.Lower.Pack r) =
    Native4d.Lower.convert ~constants:t.constants
      ~constant_store:t.constant_store src
    |> fail (fun e -> Native4d_failure.Convert e)
  in
  let g = Native4d.Lower.graph r in
  (* The destination's deferred exports exist only symbolically until resolved
     from the archive; the bundle and the reference must see the same map. *)
  let* store, _ =
    Const_ssa_materialize.materialize
      (Native_interp.capture_resolver archive)
      r.Native4d.Lower.constant_store
    |> fail (fun e -> Native4d_failure.Constants e)
  in
  let constants =
    Tensor_id.Map.union
      (fun _ _ planned -> Some planned)
      r.Native4d.Lower.constants
      (Constant_store.materialized store)
  in
  let* bundle =
    Loop_native4d.build ~config g |> fail (fun e -> Native4d_failure.Bundle e)
  in
  Err.return
    {
      bundle;
      native_graph = None;
      constants;
      reference =
        (fun inputs ->
          Native4d.Eval_direct4.run g
            ~constants:(Tensor_id.Map.bindings constants)
            ~inputs
          |> fail (fun e -> Native4d_failure.Reference e));
    }

let load dialect ~config archive =
  match dialect with
  | Dialect.Native -> native ~config archive
  | Dialect.Native4d -> native4d ~config archive
