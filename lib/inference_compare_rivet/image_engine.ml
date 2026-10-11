(* The direct assembly route as a graph engine: every invocation of the body's
   Loop IR bundle is lowered to generic Machine IR, selected for AArch64,
   allocated and framed, and assembled by Rivet into an image of its own that
   is loaded once and called on every run. A refusal at selection, allocation
   or assembly is the engine's own; no other route runs in its place. *)

module M = Machine_model.Mir_model
module Route = Machine_rivet_aarch64.Rivet_a64_route

module type CONFIG = sig
  val allocation : Route.Allocation.t
  val runtime : Machine_rivet_aarch64.Rivet_a64_runtime.t
end

module Make (X : CONFIG) = struct
  type op = Graph_ir.op
  type compiled = M.t

  let compile (g : Graph_ir.graph) =
    match Loop_ir.Loop_bundle.build g with
    | Error e ->
        Error
          (Format.asprintf "%a" Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
    | Ok bundle -> (
        match
          M.prepare
            ~route:(Route.route ~allocation:X.allocation ~runtime:X.runtime ())
            ~pipeline:Ssa_backends.Pipeline.Exact bundle
        with
        | Ok m -> Ok m
        | Error refusals ->
            Error
              (Format.asprintf "%a"
                 Fmt.(list ~sep:(any "; ") M.Refusal.pp)
                 refusals))

  let execute m (_ : Graph_ir.graph) ~inputs =
    match M.Context.create m ~constants:(fun _ -> None) with
    | Error s -> Error (Format.asprintf "%a" M.Stop.pp s)
    | Ok ctx -> (
        match
          M.Context.run ctx ~inputs:(fun id -> List.assoc_opt id inputs)
        with
        | Ok outputs -> Ok outputs
        | Error s -> Error (Format.asprintf "%a" M.Stop.pp s))

  let release (_ : M.t) = ()
end

include Make (struct
  let allocation = Route.Allocation.Reference
  let runtime = Machine_rivet_aarch64.Rivet_a64_runtime.Dependency_free
end)
