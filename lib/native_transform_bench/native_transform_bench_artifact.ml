(* The deterministic comparison artifact: one text dump per
   model, covering everything a performance change to [Rewrite]/[Pass] could
   silently perturb. Reuses [Graph_ir.pp] (the existing test-grade structural
   dump) for the graph itself, but NOT [Const_ssa.pp]/[Constant_store.pp]:
   their [pp_leaf] abbreviates a literal or opaque-materialized tensor to the
   bare word "literal"/"opaque-materialized" (const_ssa.ml), which would hide
   exactly the constant-fold miscomputation this artifact exists to catch.
   Tensor payloads instead go through [Tensor.jsont ()] with no
   [~max_elts] cap, so no value is silently truncated either. *)

module Tensor_id = Graph_ir.Tensor_id
module Run = Native_transform_bench_run

let pp_tensor_exact fmt (t : Tensor.packed) =
  match Jsont_bytesrw.encode_string (Tensor.jsont ()) t with
  | Ok s -> Fmt.string fmt s
  | Error msg -> Fmt.pf fmt "<tensor encode error: %s>" msg

let pp_constants fmt (constants : Tensor.packed Tensor_id.Map.t) =
  Fmt.pf fmt "@[<v>%a@]"
    Fmt.(
      list ~sep:cut (fun fmt (id, t) ->
          pf fmt "@[<h>%a = %a@]" Tensor_id.pp id pp_tensor_exact t))
    (Tensor_id.Map.bindings constants)

let pp_packed_fmt fmt (Payload.Fmt f) = Payload.pp_fmt fmt f

let pp_tensor_sig fmt (sg : Tensor_sig.t) =
  Fmt.pf fmt "%a %a %a" Tensor_id.pp sg.Tensor_sig.id Vec6.pp_shape
    sg.Tensor_sig.shape pp_packed_fmt sg.Tensor_sig.fmt

let pp_leaf fmt = function
  | Const_ssa.Captured c -> Fmt.pf fmt "captured %a" Const_ssa.Capture.pp c
  | Const_ssa.Literal t -> Fmt.pf fmt "literal %a" pp_tensor_exact t
  | Const_ssa.Opaque_materialized t ->
      Fmt.pf fmt "opaque-materialized %a" pp_tensor_exact t

let pp_definition fmt (id, (definition : Const_ssa.definition)) =
  match definition with
  | Const_ssa.Leaf { leaf; output } ->
      Fmt.pf fmt "@[<h>%a : %a = %a@]" Const_ssa.Value_id.pp id pp_tensor_sig
        output pp_leaf leaf
  | Const_ssa.Apply { op; output } ->
      (* [op]'s operands are already [Tensor_id.t] — the Const-SSA operand a
         [Value_id.t] names IS the tensor id it was minted from
         ([Value_id.of_tensor_id]/[to_tensor_id] are the identity coercion —
         no conversion belongs here. *)
      Fmt.pf fmt "@[<h>%a : %a = %a@]" Const_ssa.Value_id.pp id pp_tensor_sig
        output
        (Graph_ir.pp_op_with ~pp_ref:Tensor_id.pp)
        op

(* The constant plan in full — every definition's exact value, not
   [Const_ssa.pp]'s abbreviation — plus the exports (which destination tensor
   ids the plan is reachable through). *)
let pp_constant_store fmt (store : Constant_store.t) =
  Fmt.pf fmt "@[<v>exports:@,%a@,plan:@,%a@]"
    Fmt.(
      list ~sep:cut (fun fmt (tid, vid) ->
          pf fmt "@[<h>%a -> %a@]" Tensor_id.pp tid Const_ssa.Value_id.pp vid))
    (Constant_store.bindings store)
    Fmt.(list ~sep:cut pp_definition)
    (Const_ssa.bindings (Constant_store.plan store))

let pp_derived fmt (derived : (Tensor_id.t * string list) list) =
  Fmt.pf fmt "@[<v>%a@]"
    Fmt.(
      list ~sep:cut (fun fmt (id, names) ->
          pf fmt "@[<h>%a <- %a@]" Tensor_id.pp id (list ~sep:comma string)
            names))
    derived

let pp_state fmt (s : Run.state_dump) =
  Fmt.pf fmt
    "@[<v>=== graph (%s) ===@,\
     %a@,\
     @,\
     === allocator (%s) ===@,\
     %s@,\
     @,\
     === constants (%s) ===@,\
     %a@,\
     @,\
     === constant store (%s) ===@,\
     %a@]"
    s.label Graph_ir.pp s.graph s.label s.allocator_pp s.label pp_constants
    s.constants s.label pp_constant_store s.constant_store

let pp fmt (r : Run.result) =
  Fmt.pf fmt
    "@[<v>%a@,\
     @,\
     === composed map (rewrite . pack) ===@,\
     %s@,\
     @,\
     === derived (packed) ===@,\
     %a@]"
    Fmt.(list ~sep:(any "@,@,") pp_state)
    r.states r.composed_map_pp pp_derived r.derived

let to_string r = Fmt.to_to_string pp r
