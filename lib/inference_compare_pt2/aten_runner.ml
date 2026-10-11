(* The ordinary graph runner over ATen: a body's exporter nodes run in order
   through the interpreter's generated dispatch, on the handles imported for its
   inputs. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Smap = Schema_runtime.String_map

type error =
  | Aten of string
  | Closed
  | Import_mismatch of string
  | Input_inventory of string
  | Resource_exhausted
  | Value_released

let pp_error ppf = function
  | Aten m -> Fmt.pf ppf "ATen: %s" m
  | Closed -> Fmt.string ppf "prepared graph is closed"
  | Import_mismatch m -> Fmt.pf ppf "import mismatch: %s" m
  | Input_inventory m -> Fmt.pf ppf "input inventory: %s" m
  | Resource_exhausted -> Fmt.string ppf "resource exhausted"
  | Value_released -> Fmt.string ppf "value released"

type t =
  (module Core.Runner.GRAPH_RUNNER
     with type Graph.t = Aten_dialect.body
      and type Error.t = error)

let sha_digest make text =
  match
    Codecs.Payload_hasher.digest_string Inference_compare_sha256.registry
      Inference_compare_sha256.algorithm text
  with
  | Ok d -> (
      match
        make Inference_compare_sha256.algorithm ~hex:(Digest.Payload.hex d)
      with
      | Ok x -> x
      | Error _ -> invalid_arg "Aten_runner.digest")
  | Error _ -> invalid_arg "Aten_runner.digest"

let identity () : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string "aten";
    route = id Id.Engine_route.of_string "cpp-interpreter";
    build_digest =
      sha_digest Digest.Build.make
        "inference-compare-pt2/aten/cpp-interpreter/v1";
    configuration_digest =
      sha_digest Digest.Configuration.make
        "float32;cpu;dispatch=interp_dispatch";
    artifact_digest = None;
    environment_digest =
      sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ string_of_int Sys.word_size);
  }

let create () : t =
  let module R = struct
    module Graph = struct
      type t = Aten_dialect.body
    end

    module Error = struct
      type t = error

      let pp = pp_error

      let is_resource_exhaustion = function
        | Resource_exhausted -> true
        | _ -> false
    end

    module Prepared = struct
      type t = { body : Aten_dialect.body; mutable closed : bool }
    end

    module Value = struct
      type t = {
        tensor : Interp_decode.tensor;
        spec : Tensor_spec.t;
        mutable released : bool;
      }
    end

    let fail e = Err.fail ~pp_error e
    let identity = identity
    let prepare body = Ok { Prepared.body; closed = false }
    let close (p : Prepared.t) = p.closed <- true

    let import _ctx (p : Prepared.t) (spec : Tensor_spec.t) (v : L.t) =
      if p.closed then fail Closed
      else if not (Tensor_spec.equal spec (L.spec v)) then
        fail (Import_mismatch "spec")
      else
        match Conv.aten_of_logical v with
        | Ok tensor -> Ok { Value.tensor; spec; released = false }
        | Error m -> fail (Import_mismatch m)

    let export ctx (v : Value.t) =
      if v.released then fail Value_released
      else
        match Conv.logical_of_aten ctx v.spec v.tensor with
        | Ok x -> Ok x
        | Error m ->
            if m = Conv.resource then fail Resource_exhausted else fail (Aten m)

    let release_value (v : Value.t) = v.released <- true

    let run (p : Prepared.t) ~inputs =
      if p.closed then fail Closed
      else
        let body = p.body in
        let names =
          List.map
            (fun ((t : Tensor_ref.t), (v : Value.t)) ->
              (Id.Tensor_local.to_string t.local_id, v))
            inputs
        in
        if
          List.sort compare (List.map fst names)
          <> List.sort compare body.inputs
        then fail (Input_inventory "inputs differ from the graph's sources")
        else if List.exists (fun (_, (v : Value.t)) -> v.released) names then
          fail Value_released
        else
          let env =
            List.fold_left
              (fun env (n, (v : Value.t)) -> Smap.add n v.tensor env)
              Smap.empty names
          in
          let render e = Format.asprintf "%a" Interp.pp_error e in
          try
            let rec go env = function
              | [] -> Ok env
              | (_, (node : Pytorch_types.Node.t)) :: rest -> (
                  let step =
                    if Aten_model.is_assertion node then
                      Interp.assert_tensor_metadata env node
                      |> Err.map_error (fun e -> (e :> Interp.error))
                    else
                      Interp_dispatch.dispatch env node
                      |> Err.map_error (fun e -> (e :> Interp.error))
                  in
                  match step with
                  | Ok env -> go env rest
                  | Error e -> fail (Aten (render (Err.Error.kind e))))
            in
            match go env body.nodes with
            | Error e -> Error e
            | Ok env ->
                let rec outs acc = function
                  | [] -> Ok (List.rev acc)
                  | name :: rest -> (
                      match Interp_decode.resolve env name with
                      | Error e ->
                          fail
                            (Aten
                               (Format.asprintf "%a" Interp_decode.pp_error
                                  (Err.Error.kind e)))
                      | Ok tensor ->
                          let spec = List.assoc name body.model.specs in
                          let tref =
                            match
                              Aten_model.tensor_ref ~key:body.key
                                ~scope:body.scope name
                            with
                            | Ok t -> t
                            | Error m -> invalid_arg m
                          in
                          outs
                            ((tref, { Value.tensor; spec; released = false })
                            :: acc)
                            rest)
                in
                outs [] body.outputs
          with Aten_tensor.Error m -> fail (Aten m)
  end in
  (module R : Core.Runner.GRAPH_RUNNER
    with type Graph.t = Aten_dialect.body
     and type Error.t = error)
