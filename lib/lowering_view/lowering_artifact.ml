open Graph_ir

module Language = struct
  type t =
    | C
    | Coral_estimate
    | Coral_plan
    | Gnu_asm of Lowering_key.Target.t
    | Js
    | Loop
    | Machine_ir
    | Ssa
    | Wat

  let all =
    C :: Coral_estimate :: Coral_plan
    :: List.map (fun t -> Gnu_asm t) Lowering_key.Target.all
    @ [ Js; Loop; Machine_ir; Ssa; Wat ]

  let name = function
    | C -> "c"
    | Coral_estimate -> "coral_estimate"
    | Coral_plan -> "coral_plan"
    | Gnu_asm _ -> "gnu_asm"
    | Js -> "js"
    | Loop -> "loop"
    | Machine_ir -> "machine_ir"
    | Ssa -> "ssa"
    | Wat -> "wat"
end

module Refusal = struct
  type t =
    | Asm_aarch64 of Machine_rivet_aarch64_module.Rivet_a64_refusal.t
    | Asm_riscv32 of Machine_rivet_riscv32_mir_module.Rv32_mir_form.Refusal.t
    | Asm_x86_64 of Machine_rivet_x86_64_module.Rivet_x64_refusal.t
    | Coral_kernel of Coralnpu_kernels.Kernel_refusal.t
    | Coral_transfer of Coral_transfer.Transfer_plan.Refusal.t
    | Granted of Node_id.t
    | Loop_c of Loop_ir.Loop_c.error
    | Loop_lowering of Loop_ir.Loop_lower.error
    | Loop_wasm of Loop_ir.Loop_wasm.error
    | Mir_lowering of Machine_model.Mir_model.Refusal.t
    | Mir_stage of Machine_model.Mir_model.Reason.t
    | Not_a_contraction
    | Ssa_backends of string
    | Ssa_lowering of Ssa_backends.Refusal.t
    | Unknown_node of Node_id.t
    | Unsupported_family of Lowering_key.family
    | Unsupported_policy of {
        backend : Loop_ir.Loop_numerics.Backend.t;
        policy : Loop_ir.Loop_numerics.t;
      }

  let pp ppf = function
    | Asm_aarch64 r -> Machine_rivet_aarch64_module.Rivet_a64_refusal.pp ppf r
    | Asm_riscv32 r ->
        Machine_rivet_riscv32_mir_module.Rv32_mir_form.Refusal.pp ppf r
    | Asm_x86_64 r -> Machine_rivet_x86_64_module.Rivet_x64_refusal.pp ppf r
    | Coral_kernel r -> Coralnpu_kernels.Kernel_refusal.pp ppf r
    | Coral_transfer r -> Coral_transfer.Transfer_plan.Refusal.pp ppf r
    | Granted n ->
        Fmt.pf ppf
          "node %a is realized only by a grant: its bundle body is a \
           placeholder"
          Node_id.pp n
    | Loop_c e -> Loop_ir.Loop_c.pp_error ppf e
    | Loop_lowering e -> Loop_ir.Loop_lower.pp_error ppf e
    | Loop_wasm e -> Loop_ir.Loop_wasm.pp_error ppf e
    | Mir_lowering r -> Machine_model.Mir_model.Refusal.pp ppf r
    | Mir_stage r -> Machine_model.Mir_model.Reason.pp ppf r
    | Not_a_contraction ->
        Fmt.string ppf "the invocation's plan holds no matrix product"
    | Ssa_backends s -> Fmt.string ppf s
    | Ssa_lowering r -> Ssa_backends.Refusal.pp ppf r
    | Unknown_node n ->
        Fmt.pf ppf "the bundle has no invocation for %a" Node_id.pp n
    | Unsupported_family f ->
        Fmt.pf ppf "family %s is not supported" (Lowering_key.family_name f)
    | Unsupported_policy { backend; policy } ->
        Loop_ir.Loop_numerics.pp_unsupported ppf
          (`Unsupported_policy (backend, policy))
end

module Note = struct
  type t =
    | Code_bytes of int64
    | Coral_region_assumed of Coral_estimate.Coral_census.Region.t
    | Coral_transfer_budget of Coral_transfer.Transfer_units.Bytes.t
    | Fp32_refusal of Loop_ir.Loop_numerics.Refusal.t
    | Fused_multiply_adds
    | Helper_bindings of (string * string) list
    | Helpers of Loop_ir.Loop_c_runtime.Name.t list
    | Local_doubles of int64
    | Loop_target of Loop_ir.Loop_target.t
    | Mir_summary of string
    | Numerics of Loop_ir.Loop_numerics.t
    | Numerics_not_applicable of Loop_ir.Loop_numerics.t
    | Relocations of Machine_model.Mir_artifact.Relocation.t list
    | Row_blocking
    | Ssa_fp32_refusal of Ssa_ir.Ssa_numerics.Refusal.t
    | Ssa_target of Ssa_ir.Ssa_target.t
    | Symbols of Machine_model.Mir_artifact.Symbol.t list
    | Vectorization of Ssa_ir.Ssa_vectorize.report
    | Working_precision of Loop_ir.Loop_numerics.Precision.t

  let pp_tally ppf (name, (loops, work)) =
    Fmt.pf ppf "%s: %d loops, %Ld iterations" name loops work

  module Art = Machine_model.Mir_artifact

  (* One symbol as the artifact names it, with what it is. *)
  let pp_symbol ppf (s : Art.Symbol.t) =
    Fmt.pf ppf "%s (%s)" s.Art.Symbol.name
      (match s.Art.Symbol.kind with
      | Art.Symbol.Data { size; section; _ } ->
          Fmt.str "%s, %Ld bytes" (Art.Section.name section) size
      | Art.Symbol.External_function _ -> "external function"
      | Art.Symbol.Function _ -> "function"
      | Art.Symbol.Generated_attention _ -> "generated attention"
      | Art.Symbol.Generated_function _ -> "generated function")

  (* The distinct symbols relocations name, each with how many name it. *)
  let relocation_tally rs =
    List.fold_left
      (fun acc (r : Art.Relocation.t) ->
        let s = r.Art.Relocation.symbol in
        match List.assoc_opt s acc with
        | None -> acc @ [ (s, 1) ]
        | Some n -> (s, n + 1) :: List.remove_assoc s acc)
      [] rs
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)

  let pp ppf = function
    | Code_bytes n -> Fmt.pf ppf "code: %Ld bytes" n
    | Coral_region_assumed r ->
        Fmt.pf ppf "every buffer assumed in %s"
          (Coral_estimate.Coral_census.Region.name r)
    | Coral_transfer_budget b ->
        Fmt.pf ppf "DTCM budget: %d bytes"
          (Coral_transfer.Transfer_units.Bytes.to_int b)
    | Fp32_refusal r ->
        Fmt.pf ppf "binary32 refused: %a" Loop_ir.Loop_numerics.Refusal.pp r
    | Fused_multiply_adds -> Fmt.string ppf "fused multiply-adds"
    | Helper_bindings bs ->
        Fmt.pf ppf "helper bindings: %a"
          Fmt.(list ~sep:(any ", ") (pair ~sep:(any " -> ") string string))
          bs
    | Helpers hs ->
        Fmt.pf ppf "helpers: %a"
          Fmt.(list ~sep:(any ", ") string)
          (List.map Loop_ir.Loop_c_runtime.Name.to_string hs)
    | Local_doubles n -> Fmt.pf ppf "local scratch: %Ld doubles" n
    | Loop_target t -> Fmt.pf ppf "vector target: %s" t.Loop_ir.Loop_target.name
    | Mir_summary s -> Fmt.string ppf s
    | Numerics n -> Fmt.pf ppf "numerics: %s" (Loop_ir.Loop_numerics.name n)
    | Numerics_not_applicable n ->
        Fmt.pf ppf "numerics %s does not apply to this family"
          (Loop_ir.Loop_numerics.name n)
    | Relocations rs ->
        Fmt.pf ppf "relocations: %d (%a)" (List.length rs)
          Fmt.(list ~sep:(any ", ") (fun ppf (s, n) -> Fmt.pf ppf "%s x%d" s n))
          (relocation_tally rs)
    | Row_blocking -> Fmt.string ppf "row loops blocked around a vector loop"
    | Ssa_fp32_refusal r ->
        Fmt.pf ppf "binary32 refused: %a" Ssa_ir.Ssa_numerics.Refusal.pp r
    | Ssa_target t -> Fmt.pf ppf "SSA target: %s" t.Ssa_ir.Ssa_target.name
    | Symbols ss ->
        Fmt.pf ppf "symbols: %a" Fmt.(list ~sep:(any ", ") pp_symbol) ss
    | Vectorization report ->
        Fmt.pf ppf "vectorization: %a"
          Fmt.(list ~sep:(any "; ") pp_tally)
          (Ssa_ir.Ssa_vectorize.tally report)
    | Working_precision p ->
        Fmt.pf ppf "working precision: %s"
          (Loop_ir.Loop_numerics.Precision.name p)
end

type body =
  | Refused of Refusal.t
  | Text of { language : Language.t; text : string; truncated : bool }

type t = { key : Lowering_key.t; body : body; notes : Note.t list }

let pp ppf { key; body; notes } =
  Fmt.pf ppf "@[<v>%a@," Lowering_key.pp_key key;
  (match body with
  | Refused r -> Fmt.pf ppf "refused: %a" Refusal.pp r
  | Text { text; truncated; _ } ->
      Fmt.string ppf text;
      if truncated then Fmt.pf ppf "@,(truncated)");
  List.iter (fun n -> Fmt.pf ppf "@,note: %a" Note.pp n) notes;
  Fmt.pf ppf "@]"
