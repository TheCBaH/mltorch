open Graph_ir

module Language = struct
  type t = C | Js | Loop | Ssa | Wat

  let all = [ C; Js; Loop; Ssa; Wat ]

  let name = function
    | C -> "c"
    | Js -> "js"
    | Loop -> "loop"
    | Ssa -> "ssa"
    | Wat -> "wat"
end

module Refusal = struct
  type t =
    | Granted of Node_id.t
    | Loop_c of Loop_ir.Loop_c.error
    | Loop_lowering of Loop_ir.Loop_lower.error
    | Loop_wasm of Loop_ir.Loop_wasm.error
    | Ssa_backends of string
    | Ssa_lowering of Ssa_backends.Refusal.t
    | Unknown_node of Node_id.t
    | Unsupported_family of Lowering_key.family
    | Unsupported_policy of {
        backend : Loop_ir.Loop_numerics.Backend.t;
        policy : Loop_ir.Loop_numerics.t;
      }

  let pp ppf = function
    | Granted n ->
        Fmt.pf ppf
          "node %a is realized only by a grant: its bundle body is a \
           placeholder"
          Node_id.pp n
    | Loop_c e -> Loop_ir.Loop_c.pp_error ppf e
    | Loop_lowering e -> Loop_ir.Loop_lower.pp_error ppf e
    | Loop_wasm e -> Loop_ir.Loop_wasm.pp_error ppf e
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
    | Fp32_refusal of Loop_ir.Loop_numerics.Refusal.t
    | Fused_multiply_adds
    | Helpers of Loop_ir.Loop_c_runtime.Name.t list
    | Local_doubles of int64
    | Loop_target of Loop_ir.Loop_target.t
    | Numerics of Loop_ir.Loop_numerics.t
    | Numerics_not_applicable of Loop_ir.Loop_numerics.t
    | Row_blocking
    | Ssa_fp32_refusal of Ssa_ir.Ssa_numerics.Refusal.t
    | Ssa_target of Ssa_ir.Ssa_target.t
    | Vectorization of Ssa_ir.Ssa_vectorize.report
    | Working_precision of Loop_ir.Loop_numerics.Precision.t

  let pp_tally ppf (name, (loops, work)) =
    Fmt.pf ppf "%s: %d loops, %Ld iterations" name loops work

  let pp ppf = function
    | Fp32_refusal r ->
        Fmt.pf ppf "binary32 refused: %a" Loop_ir.Loop_numerics.Refusal.pp r
    | Fused_multiply_adds -> Fmt.string ppf "fused multiply-adds"
    | Helpers hs ->
        Fmt.pf ppf "helpers: %a"
          Fmt.(list ~sep:(any ", ") string)
          (List.map Loop_ir.Loop_c_runtime.Name.to_string hs)
    | Local_doubles n -> Fmt.pf ppf "local scratch: %Ld doubles" n
    | Loop_target t -> Fmt.pf ppf "vector target: %s" t.Loop_ir.Loop_target.name
    | Numerics n -> Fmt.pf ppf "numerics: %s" (Loop_ir.Loop_numerics.name n)
    | Numerics_not_applicable n ->
        Fmt.pf ppf "numerics %s does not apply to this family"
          (Loop_ir.Loop_numerics.name n)
    | Row_blocking -> Fmt.string ppf "row loops blocked around a vector loop"
    | Ssa_fp32_refusal r ->
        Fmt.pf ppf "binary32 refused: %a" Ssa_ir.Ssa_numerics.Refusal.pp r
    | Ssa_target t -> Fmt.pf ppf "SSA target: %s" t.Ssa_ir.Ssa_target.name
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
