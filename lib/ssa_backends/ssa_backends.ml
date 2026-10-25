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

(* The program and the working precision its pipeline chose. A kernel's
   precision is the plan's, not whatever types its text happens to use: the
   strict pipelines hold binary32 buffers' values in binary32 but compute in
   binary64, and must not report as binary32 kernels. *)
let prepare pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  match
    Err.payload (Ssa_lower.Ssa_lower_plan.lower inv.Loop_ir.Loop_bundle.placed)
  with
  | Error (`Unsupported u) ->
      Error (Fmt.str "%a" Ssa_lower.Ssa_unsupported.pp u)
  | Ok p -> (
      match pipeline with
      | Pipeline.Representation -> Ok (p, Loop_ir.Loop_numerics.Precision.F64)
      | Pipeline.Exact ->
          Ok
            ( fst (Ssa_opt.run ~alias ~passes:exact_passes p),
              Loop_ir.Loop_numerics.Precision.F64 )
      | Pipeline.Planned { numerics; target } ->
          Result.map_error
            (fun r -> Fmt.str "%a" Ssa_numerics.Refusal.pp r)
            (Result.map
               (fun (plan : Ssa_plan.t) ->
                 ( plan.Ssa_plan.program,
                   match plan.Ssa_plan.precision with
                   | Ssa_numerics.Precision.F32 ->
                       Loop_ir.Loop_numerics.Precision.F32
                   | Ssa_numerics.Precision.F64 ->
                       Loop_ir.Loop_numerics.Precision.F64 ))
               (Ssa_plan.admitted (Ssa_plan.resolve ~target ~alias ~numerics p)))
      )

let program pipeline inv = Result.map fst (prepare pipeline inv)

(* The resolved plan of an invocation under a policy and a target: what a
   consumer that needs the plan's own record (its numerics, precision and the
   target it was made for) takes instead of only the program. *)
let plan ?contractions ?bf16_pairs ~numerics ~target inv =
  match
    Err.payload (Ssa_lower.Ssa_lower_plan.lower inv.Loop_ir.Loop_bundle.placed)
  with
  | Error (`Unsupported u) ->
      Error (Fmt.str "%a" Ssa_lower.Ssa_unsupported.pp u)
  | Ok p ->
      Result.map_error
        (fun r -> Fmt.str "%a" Ssa_numerics.Refusal.pp r)
        (Ssa_plan.admitted
           (Ssa_plan.resolve ~target ~alias ?contractions ?bf16_pairs ~numerics
              p))

let blocked ~group inv =
  Result.map
    (fun p ->
      let q, report =
        Ssa_opt.run ~alias
          ~passes:
            [
              Ssa_opt.block ~alias ~group:(Ssa_opt_block.Fixed group);
              Ssa_opt.share ~alias;
              Ssa_opt.simplify;
            ]
          p
      in
      if Option.value ~default:0 (List.assoc_opt "block" report) > 0 then Some q
      else None)
    (program Pipeline.Exact inv)

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
           "%a (%s) is touched by the SSA program and is no buffer of the \
            invocation: a fusion scratch the pipeline did not remove"
           Ssa_id.Buffer.pp n.Ssa_buffer.id
           (Ssa_buffer.role_name n.Ssa_buffer.role))

let sites (inv : Loop_ir.Loop_bundle.invocation) =
  Loop_ir.Loop_js_failure.sites inv.Loop_ir.Loop_bundle.program

let c ~pipeline ~name inv =
  let* p, precision = prepare pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_c.arguments in
  match Ssa_c.kernel ~buffers ~sites:(sites inv) ~name p with
  | Ok (k, _) -> Ok { k with Loop_ir.Loop_c.precision }
  | Error e -> Error (Fmt.str "%a" Ssa_c.pp_error e)

let wasm ~pipeline ~table_alloc inv =
  let* p, precision = prepare pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_wasm.arguments in
  let relaxed_madd =
    match pipeline with
    | Pipeline.Planned { target; _ } -> target.Ssa_target.relaxed_madd
    | Pipeline.Exact | Pipeline.Representation -> false
  in
  match
    Ssa_wasm.kernel ~buffers ~sites:(sites inv) ~relaxed_madd ~table_alloc p
  with
  | Ok k -> Ok { k with Loop_ir.Loop_wasm.precision }
  | Error e -> Error (Fmt.str "%a" Ssa_wasm.pp_error e)

let js ~pipeline inv =
  let* p = program pipeline inv in
  let* buffers = arguments inv p ~named:Ssa_js.arguments in
  match Ssa_js.program ~buffers ~sites:(sites inv) p with
  | Ok (ast, _) -> Ok ast
  | Error e -> Error (Fmt.str "%a" Ssa_js.pp_error e)

module Coverage = struct
  type column =
    | Exact_f32 of Ssa_target.t
    | Loop_fp32_ordered of Loop_ir.Loop_target.t
    | Narrowed_f32 of Ssa_target.t

  let name = function
    | Exact_f32 t -> "ssa exact " ^ t.Ssa_target.name
    | Loop_fp32_ordered t -> "loop fp32_ordered " ^ t.Loop_ir.Loop_target.name
    | Narrowed_f32 t -> "ssa narrowed " ^ t.Ssa_target.name

  let columns =
    List.concat_map
      (fun (s, l) -> [ Narrowed_f32 s; Exact_f32 s; Loop_fp32_ordered l ])
      [
        (Ssa_target.neon128, Loop_ir.Loop_target.neon128);
        (Ssa_target.wasm128, Loop_ir.Loop_target.wasm128);
      ]

  type tally = (string * (int * int64)) list

  let merge (a : tally) (b : tally) : tally =
    List.fold_left
      (fun acc (k, (n, w)) ->
        let m, x = Option.value ~default:(0, 0L) (List.assoc_opt k acc) in
        (k, (m + n, Int64.add w x)) :: List.remove_assoc k acc)
      a b
    |> List.sort (fun (x, _) (y, _) -> String.compare x y)

  let blocked_by_exact_access = "(narrowed vectorizes, exact access refuses)"

  let lowered (inv : Loop_ir.Loop_bundle.invocation) =
    match
      Err.payload
        (Ssa_lower.Ssa_lower_plan.lower inv.Loop_ir.Loop_bundle.placed)
    with
    | Error (`Unsupported u) ->
        Error (Fmt.str "%a" Ssa_lower.Ssa_unsupported.pp u)
    | Ok p -> Ok p

  let vectorize ~target ~exact inv =
    let* p = lowered inv in
    Result.map_error
      (fun r -> Fmt.str "%a" Ssa_numerics.Refusal.pp r)
      (Ssa_plan.vectorize_f32 ~alias
         ?elements:(if exact then Some [ Ssa_type.F32 ] else None)
         ~target ~exact p)

  let report ~target ~exact inv = Result.map snd (vectorize ~target ~exact inv)
  let iotas_key = "(iota instructions)"

  let rec iotas (r : Ssa_region.t) =
    List.fold_left
      (fun n (s : Ssa_region.t Ssa_stmt.t) ->
        match s with
        | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Vec_iota _; _ } -> n + 1
        | Ssa_stmt.Instr _ -> n
        | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
            n + iotas body
        | Ssa_stmt.If { then_; else_; _ } -> n + iotas then_ + iotas else_)
      0 r.Ssa_region.body

  let vectorized (d : Ssa_vectorize.Decision.t) =
    match d.Ssa_vectorize.Decision.outcome with
    | Ssa_vectorize.Decision.Vectorized -> true
    | Ssa_vectorize.Decision.Kept_scalar _ -> false

  let at_exact_access (d : Ssa_vectorize.Decision.t) =
    match d.Ssa_vectorize.Decision.outcome with
    | Ssa_vectorize.Decision.Kept_scalar
        (Ssa_vectorize.Reason.No_vector_form
           ( Ssa_op.Load_in_bounds { decode = Ssa_op.Decode.F32_exact; _ }
           | Ssa_op.Store { encode = Ssa_op.Encode.F32_exact; _ } )) ->
        true
    | Ssa_vectorize.Decision.Kept_scalar _ | Ssa_vectorize.Decision.Vectorized
      ->
        false

  (* The loops the narrowed form vectorizes and the exact form keeps scalar at
     a binary32 access: what accepting those accesses alone would add. *)
  let exact_blocked ~narrowed ~exact =
    List.filter
      (fun (d : Ssa_vectorize.Decision.t) ->
        vectorized d
        && List.exists
             (fun (e : Ssa_vectorize.Decision.t) ->
               Ssa_id.Region.equal e.Ssa_vectorize.Decision.loop
                 d.Ssa_vectorize.Decision.loop
               && at_exact_access e)
             exact)
      narrowed
    |> List.fold_left
         (fun (n, w) (d : Ssa_vectorize.Decision.t) ->
           ( n + 1,
             Int64.add w
               (Int64.mul d.Ssa_vectorize.Decision.trips
                  d.Ssa_vectorize.Decision.executions) ))
         (0, 0L)

  let tally column (inv : Loop_ir.Loop_bundle.invocation) =
    match column with
    | Exact_f32 target ->
        let* narrowed = report ~target ~exact:false inv in
        let* exact = report ~target ~exact:true inv in
        let n, w = exact_blocked ~narrowed ~exact in
        let t = Ssa_vectorize.tally exact in
        Ok (if n = 0 then t else merge t [ (blocked_by_exact_access, (n, w)) ])
    | Narrowed_f32 target ->
        let* q, r = vectorize ~target ~exact:false inv in
        let n = iotas q.Ssa_program.entry in
        let t = Ssa_vectorize.tally r in
        Ok (if n = 0 then t else merge t [ (iotas_key, (n, 0L)) ])
    | Loop_fp32_ordered target ->
        let plan =
          Loop_ir.Loop_plan.resolve ~target
            ~numerics:Loop_ir.Loop_numerics.Fp32_ordered
            inv.Loop_ir.Loop_bundle.program
        in
        Ok (Loop_ir.Loop_vectorize.tally plan.Loop_ir.Loop_plan.report)

  let bundle columns (b : Loop_ir.Loop_bundle.t) =
    List.map
      (fun c ->
        let t, refused =
          List.fold_left
            (fun (t, refused) inv ->
              match tally c inv with
              | Ok u -> (merge t u, refused)
              | Error _ -> (t, refused + 1))
            ([], 0) b.Loop_ir.Loop_bundle.invocations
        in
        (c, t, refused))
      columns

  let pp_tally ppf (t : tally) =
    Fmt.(list ~sep:(any ", ") (fun ppf (k, (n, w)) -> pf ppf "%s %d/%Ld" k n w))
      ppf t
end
