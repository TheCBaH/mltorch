(* The ordinary graph runner over Native: a body is a graph, run by the direct
   evaluator on the tensors imported for its inputs. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Tid = Graph_ir.Tensor_id

type error =
  | Closed
  | Import_mismatch of string
  | Input_inventory of string
  | Native of string
  | Resource_exhausted
  | Value_released

let pp_error ppf = function
  | Closed -> Fmt.string ppf "prepared graph is closed"
  | Import_mismatch m -> Fmt.pf ppf "import mismatch: %s" m
  | Input_inventory m -> Fmt.pf ppf "input inventory: %s" m
  | Native m -> Fmt.pf ppf "Native: %s" m
  | Resource_exhausted -> Fmt.string ppf "resource exhausted"
  | Value_released -> Fmt.string ppf "value released"

type t =
  (module Core.Runner.GRAPH_RUNNER
     with type Graph.t = Native_dialect.body
      and type Error.t = error)

(* [config] names the storage and scheduling the evaluator runs under, when it
   is not the default. *)
let identity ?artifact ?(dots = Direct.Binary64) ?config () : Engine_identity.t
    =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string "native";
    route = id Id.Engine_route.of_string "eval-direct";
    build_digest =
      Aten_runner.sha_digest Digest.Build.make
        "inference-compare-pt2/native/eval-direct/v1";
    configuration_digest =
      Aten_runner.sha_digest Digest.Configuration.make
        ((match dots with
           | Direct.Binary64 -> "dot-accumulation=binary64;float-to-int=checked"
           | Direct.Binary32_sequential ->
               "dot-accumulation=binary32-sequential;float-to-int=checked")
        ^ match config with None -> "" | Some c -> ";" ^ c);
    artifact_digest = artifact;
    environment_digest =
      Aten_runner.sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ string_of_int Sys.word_size);
  }

let create ?artifact ?(dots = Direct.Binary64) () : t =
  let module R = struct
    module Graph = struct
      type t = Native_dialect.body
    end

    module Error = struct
      type t = error

      let pp = pp_error

      let is_resource_exhaustion = function
        | Resource_exhausted -> true
        | _ -> false
    end

    module Prepared = struct
      type t = { body : Native_dialect.body; mutable closed : bool }
    end

    module Value = struct
      type t = {
        packed : Tensor.packed;
        spec : Tensor_spec.t;
        mutable released : bool;
      }
    end

    let fail e = Err.fail ~pp_error e
    let identity () = identity ?artifact ~dots ()
    let prepare body = Ok { Prepared.body; closed = false }
    let close (p : Prepared.t) = p.closed <- true

    let import _ctx (p : Prepared.t) (spec : Tensor_spec.t) (v : L.t) =
      if p.closed then fail Closed
      else if not (Tensor_spec.equal spec (L.spec v)) then
        fail (Import_mismatch "spec")
      else
        match Conv.native_of_logical v with
        | Ok packed -> Ok { Value.packed; spec; released = false }
        | Error m -> fail (Import_mismatch m)

    let export ctx (v : Value.t) =
      if v.released then fail Value_released
      else
        match Conv.logical_of_native ctx v.spec v.packed with
        | Ok x -> Ok x
        | Error m ->
            if m = Conv.resource then fail Resource_exhausted
            else fail (Native m)

    let release_value (v : Value.t) = v.released <- true

    let run (p : Prepared.t) ~inputs =
      if p.closed then fail Closed
      else
        let body = p.body in
        let id_of (t : Tensor_ref.t) =
          let s = Id.Tensor_local.to_string t.local_id in
          Option.map Tid.of_int
            (int_of_string_opt (String.sub s 1 (String.length s - 1)))
        in
        let bound = List.map (fun (t, (v : Value.t)) -> (id_of t, v)) inputs in
        let wanted = List.sort Tid.compare body.graph.inputs in
        let got = List.sort Tid.compare (List.filter_map fst bound) in
        if
          List.exists (fun (i, _) -> i = None) bound
          || not (List.equal Tid.equal wanted got)
        then fail (Input_inventory "inputs differ from the graph's sources")
        else if List.exists (fun (_, (v : Value.t)) -> v.released) bound then
          fail Value_released
        else
          let inputs =
            List.map (fun (i, (v : Value.t)) -> (Option.get i, v.packed)) bound
          in
          match
            Direct.with_dot_accumulation dots (fun () ->
                Eval_direct.run body.graph ~inputs)
          with
          | Error e ->
              fail
                (Native
                   (Format.asprintf "%a" Eval_direct.pp_error (Err.Error.kind e)))
          | Ok env ->
              let rec outs acc = function
                | [] -> Ok (List.rev acc)
                | id :: rest -> (
                    match Tid.Map.find_opt id env with
                    | None -> fail (Native "a declared output was not produced")
                    | Some packed -> (
                        let local = Printf.sprintf "t%d" (Tid.to_int id) in
                        let tref =
                          match
                            Native_model.tensor_ref ~key:body.key
                              ~scope:body.scope id
                          with
                          | Ok t -> t
                          | Error m -> invalid_arg (m ^ local)
                        in
                        match Native_model.sig_of body.graph id with
                        | Error m -> fail (Native m)
                        | Ok sg -> (
                            match Native_model.spec_of_sig sg with
                            | Error m -> fail (Native m)
                            | Ok spec ->
                                outs
                                  ( { Value.packed; spec; released = false }
                                  |> fun v -> (tref, v) :: acc )
                                  rest)))
              in
              outs [] body.graph.outputs
  end in
  (module R : Core.Runner.GRAPH_RUNNER
    with type Graph.t = Native_dialect.body
     and type Error.t = error)
