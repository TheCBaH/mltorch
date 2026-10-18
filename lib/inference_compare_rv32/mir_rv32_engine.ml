(* The RV32 Machine IR route as a graph engine: every invocation of the body's
   Loop IR bundle planned binary32 for the scalar target, selected onto ILP32,
   allocated, framed, published and built by Rivet into a static ELF, each call
   a process under qemu-riscv32 with D disabled. A refusal at any step is the
   engine's own; no other route runs in its place. *)

module M = Machine_model.Mir_model
module Route = Machine_rivet_riscv32_mir.Rv32_mir_route

type op = Graph_ir.op
type compiled = M.t

let pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let compile (g : Graph_ir.graph) =
  match Loop_ir.Loop_bundle.build g with
  | Error e ->
      Error
        (Format.asprintf "%a" Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
  | Ok bundle -> (
      match M.prepare ~route:(Route.route ()) ~pipeline bundle with
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
      match M.Context.run ctx ~inputs:(fun id -> List.assoc_opt id inputs) with
      | Ok outputs -> Ok outputs
      | Error s -> Error (Format.asprintf "%a" M.Stop.pp s))

let release (_ : M.t) = ()
