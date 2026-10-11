(* The ordinary graph runner over any Native-family engine: a body is a graph,
   executed by the engine on the tensors imported for its inputs. The engine
   supplies only how to compile and run a graph; value import and export, the
   input inventory and ownership are the same for every engine. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Tid = Graph_ir.Tensor_id
module G = Graph_model

type error =
  | Closed
  | Engine of string
  | Import_mismatch of string
  | Input_inventory of string
  | Resource_exhausted
  | Value_released

let pp_error ppf = function
  | Closed -> Fmt.string ppf "prepared graph is closed"
  | Engine m -> Fmt.pf ppf "engine: %s" m
  | Import_mismatch m -> Fmt.pf ppf "import mismatch: %s" m
  | Input_inventory m -> Fmt.pf ppf "input inventory: %s" m
  | Resource_exhausted -> Fmt.string ppf "resource exhausted"
  | Value_released -> Fmt.string ppf "value released"

module type ENGINE = sig
  type op
  type compiled

  val compile : op Graph_common.Graph.t -> (compiled, string) result
  (** Prepares a body for execution; a refusal is the engine's own and is
      reported, never replaced by another route. *)

  val execute :
    compiled ->
    op Graph_common.Graph.t ->
    inputs:(Tid.t * Tensor.packed) list ->
    (Tensor.packed list, string) result
  (** The body's outputs in graph output order. *)

  val release : compiled -> unit
end

type 'body t =
  (module Core.Runner.GRAPH_RUNNER
     with type Graph.t = 'body
      and type Error.t = error)

let sha_digest = Aten_runner.sha_digest

module Make (X : sig
  type op

  include ENGINE with type op := op
end) =
struct
  type body = X.op Graph_body.t

  let create ~(identity : unit -> Engine_identity.t) : body t =
    let module R = struct
      module Graph = struct
        type t = body
      end

      module Error = struct
        type t = error

        let pp = pp_error

        let is_resource_exhaustion = function
          | Resource_exhausted -> true
          | _ -> false
      end

      module Prepared = struct
        type t = { body : body; compiled : X.compiled; mutable closed : bool }
      end

      module Value = struct
        type t = {
          packed : Tensor.packed;
          spec : Tensor_spec.t;
          mutable released : bool;
        }
      end

      let fail e = Err.fail ~pp_error e
      let identity = identity

      let prepare (body : body) =
        match X.compile body.graph with
        | Ok compiled -> Ok { Prepared.body; compiled; closed = false }
        | Error m -> fail (Engine m)

      let close (p : Prepared.t) =
        if not p.closed then (
          p.closed <- true;
          X.release p.compiled)

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
              else fail (Engine m)

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
          let bound =
            List.map (fun (t, (v : Value.t)) -> (id_of t, v)) inputs
          in
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
              List.map
                (fun (i, (v : Value.t)) -> (Option.get i, v.packed))
                bound
            in
            match X.execute p.compiled body.graph ~inputs with
            | Error m -> fail (Engine m)
            | Ok outputs ->
                if List.length outputs <> List.length body.graph.outputs then
                  fail (Engine "a declared output was not produced")
                else
                  let rec outs acc = function
                    | [] -> Ok (List.rev acc)
                    | (id, packed) :: rest -> (
                        match
                          G.tensor_ref ~key:body.key ~scope:body.scope id
                        with
                        | Error m -> fail (Engine m)
                        | Ok tref -> (
                            match G.spec_of body.graph id with
                            | Error m -> fail (Engine m)
                            | Ok spec ->
                                outs
                                  (( tref,
                                     { Value.packed; spec; released = false } )
                                  :: acc)
                                  rest))
                  in
                  outs [] (List.combine body.graph.outputs outputs)
    end in
    (module R : Core.Runner.GRAPH_RUNNER
      with type Graph.t = body
       and type Error.t = error)
end
