open Ssa_ir

module Pipeline = struct
  type t =
    | Exact
    | Planned of { numerics : Ssa_numerics.t; target : Ssa_target.t }
    | Representation

  let name = function
    | Exact -> "exact"
    | Planned { numerics; target } ->
        Fmt.str "planned:%s:%s"
          (Ssa_numerics.name numerics)
          target.Ssa_target.name
    | Representation -> "representation"
end

let ( let* ) = Result.bind

(* One invocation's buffers are distinct memory: the storage plan allocates every
   output before it releases any operand, so no two buffers an invocation names
   are live in the same cell (the arena checker rejects an overlap), and the
   scratch region its locals use is separate again. This is the guarantee
   [Distinct_buffers] asks a caller to have established; the Loop emitters rely
   on the same fact when they vectorize. *)
let alias = Ssa_effects.Distinct_buffers

let exact_passes =
  [
    Ssa_opt.simplify;
    Ssa_opt.guards;
    Ssa_opt.simplify;
    Ssa_opt.hoist ~alias;
    Ssa_opt.share ~alias;
    Ssa_opt.simplify;
  ]

let program pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  match
    Err.payload (Ssa_lower.Ssa_lower_plan.lower inv.Loop_ir.Loop_bundle.placed)
  with
  | Error (`Unsupported u) ->
      Error (Fmt.str "%a" Ssa_lower.Ssa_unsupported.pp u)
  | Ok p -> (
      match pipeline with
      | Pipeline.Representation -> Ok p
      | Pipeline.Exact -> Ok (fst (Ssa_opt.run ~alias ~passes:exact_passes p))
      | Pipeline.Planned { numerics; target } ->
          Ok (Ssa_plan.resolve ~target ~alias ~numerics p).Ssa_plan.program)

(* The invocation's buffers, as SSA declares them, in the program's order: the
   bundle binds them positionally to edges. A buffer the SSA program names that
   the invocation does not have cannot be bound. *)
let arguments (inv : Loop_ir.Loop_bundle.invocation) (p : Ssa_program.t) ~named
    =
  let ssa_buffer (b : Loop_ir.Loop_buffer.t) =
    let id = Ssa_id.Buffer.of_int (Tensor_id.to_int b.Loop_ir.Loop_buffer.id) in
    match Ssa_program.find_buffer p id with
    | Some sb -> Ok sb
    | None ->
        Error
          (Fmt.str "t%d is a buffer of the invocation the SSA program lacks"
             (Tensor_id.to_int b.Loop_ir.Loop_buffer.id))
  in
  let* buffers =
    List.fold_right
      (fun b acc ->
        let* acc = acc in
        let* sb = ssa_buffer b in
        Ok (sb :: acc))
      inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers (Ok [])
  in
  match
    List.find_opt
      (fun (n : Ssa_buffer.t) ->
        not
          (List.exists
             (fun (b : Ssa_buffer.t) ->
               Ssa_id.Buffer.equal b.Ssa_buffer.id n.Ssa_buffer.id)
             buffers))
      (named p)
  with
  | None -> Ok buffers
  | Some n ->
      Error
        (Fmt.str
           "%a is touched by the SSA program and is no buffer of the invocation"
           Ssa_id.Buffer.pp n.Ssa_buffer.id)

let sites (inv : Loop_ir.Loop_bundle.invocation) =
  Loop_ir.Loop_js_failure.sites inv.Loop_ir.Loop_bundle.program

let c ~pipeline ~name inv =
  let* p = program pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_c.arguments in
  match Ssa_c.kernel ~buffers ~sites:(sites inv) ~name p with
  | Ok (k, _) -> Ok k
  | Error e -> Error (Fmt.str "%a" Ssa_c.pp_error e)

let wasm ~pipeline ~table_alloc inv =
  let* p = program pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_wasm.arguments in
  let relaxed_madd =
    match pipeline with
    | Pipeline.Planned { target; _ } -> target.Ssa_target.relaxed_madd
    | Pipeline.Exact | Pipeline.Representation -> false
  in
  match
    Ssa_wasm.kernel ~buffers ~sites:(sites inv) ~relaxed_madd ~table_alloc p
  with
  | Ok k -> Ok k
  | Error e -> Error (Fmt.str "%a" Ssa_wasm.pp_error e)

let js ~pipeline inv =
  let* p = program pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_js.arguments in
  match Ssa_js.program ~buffers ~sites:(sites inv) p with
  | Ok (ast, _) -> Ok ast
  | Error e -> Error (Fmt.str "%a" Ssa_js.pp_error e)
