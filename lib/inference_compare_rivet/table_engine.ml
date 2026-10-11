(* The table host as a graph engine: the same Machine IR code as the image
   route, but bound through a table of addresses, so a call reads and writes
   the context's own tensors and nothing is copied. A refusal while preparing
   any invocation is the engine's own. *)

module H = Machine_rivet_aarch64.Rivet_a64_host
module M = Machine_model.Mir_model
module Route = Machine_rivet_aarch64.Rivet_a64_route

module type CONFIG = sig
  val allocation : Route.Allocation.t
  val runtime : Machine_rivet_aarch64.Rivet_a64_runtime.t
end

module Make (X : CONFIG) = struct
  type op = Graph_ir.op
  type compiled = H.t

  let compile (g : Graph_ir.graph) =
    match Loop_ir.Loop_bundle.build g with
    | Error e ->
        Error
          (Format.asprintf "%a" Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
    | Ok bundle -> (
        match
          H.prepare ~allocation:X.allocation ~runtime:X.runtime
            ~pipeline:Ssa_backends.Pipeline.Exact bundle
        with
        | Ok h -> Ok h
        | Error refusals ->
            Error
              (Format.asprintf "%a"
                 Fmt.(list ~sep:(any "; ") M.Refusal.pp)
                 refusals))

  let execute h (_ : Graph_ir.graph) ~inputs =
    match H.Context.create h ~constants:(fun _ -> None) with
    | Error s -> Error (Format.asprintf "%a" M.Stop.pp s)
    | Ok cx -> (
        match H.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) with
        | Ok outs -> Ok outs
        | Error s -> Error (Format.asprintf "%a" M.Stop.pp s))

  let release (_ : H.t) = ()
end
