open Loop_ir
module Artifact = Lowering_artifact
module Summary = Lowering_summary
module Language = Artifact.Language
module Refusal = Artifact.Refusal
module Note = Artifact.Note
module K = Lowering_key

let kernel_name = "kernel"
let ( let* ) = Result.bind

(* What a family produced: its text and the facts the text does not show. *)
type made = Language.t * string * Note.t list

(* The policy a family runs when the key names none: what the drivers run. *)
let default_numerics : Language.t -> Loop_numerics.t = function
  | Language.Js -> Loop_numerics.Reference_f64
  | Language.C | Language.Coral_estimate | Language.Coral_plan
  | Language.Gnu_asm _ | Language.Loop | Language.Machine_ir | Language.Ssa
  | Language.Wat ->
      Loop_numerics.Simd_fp32_relaxed

let numerics_of family = function
  | Some n -> n
  | None -> default_numerics family

(* JavaScript runs no fp32 policy. *)
let js_policy = function
  | None -> Ok ()
  | Some n -> (
      match
        Err.payload
          (Loop_numerics.check ~backend:Loop_numerics.Backend.Javascript n)
      with
      | Ok _ -> Ok ()
      | Error (`Unsupported_policy (backend, policy)) ->
          Error (Refusal.Unsupported_policy { backend; policy }))

(* The Loop route's program: the invocation's own when every pass ran (what the
   bundle's builders emit), else its placed kernel lowered raw and run through
   the subset, the way [Loop_node_program.lower_placed] selects one. *)
let loop_program (spec : K.Loop_spec.t) (inv : Loop_bundle.invocation) =
  if spec.K.Loop_spec.passes = Loop_opt.Pass.all then Ok inv.Loop_bundle.program
  else
    match Err.payload (Loop_lower.lower_unoptimized inv.Loop_bundle.placed) with
    | Error e -> Error (Refusal.Loop_lowering e)
    | Ok raw ->
        Ok (Loop_opt.run ~passes:(Loop_opt.select spec.K.Loop_spec.passes) raw)

let loop_text p = Fmt.str "%a" Loop_pp.program p

(* A policy other than the reference one vectorizes, as every driver does. *)
let loop_vector family numerics target =
  let n = numerics_of family numerics in
  (n, if n = Loop_numerics.Reference_f64 then None else Some target)

let vector_notes numerics vector =
  Note.Numerics numerics
  :: (match vector with None -> [] | Some t -> [ Note.Loop_target t ])

let c_notes (k : Loop_c_kernel.t) =
  Note.Working_precision k.Loop_c_kernel.precision
  :: List.concat
       [
         (match k.Loop_c_kernel.refusal with
         | None -> []
         | Some r -> [ Note.Fp32_refusal r ]);
         (match k.Loop_c_kernel.helpers with
         | [] -> []
         | hs -> [ Note.Helpers hs ]);
         (if Int64.compare k.Loop_c_kernel.local_doubles 0L > 0 then
            [ Note.Local_doubles k.Loop_c_kernel.local_doubles ]
          else []);
       ]

let c_text (k : Loop_c_kernel.t) =
  if k.Loop_c_kernel.prelude = "" then k.Loop_c_kernel.source
  else k.Loop_c_kernel.prelude ^ "\n" ^ k.Loop_c_kernel.source

let loop_family spec inv : (made, Refusal.t) result =
  let* p = loop_program spec inv in
  Ok
    ( Language.Loop,
      loop_text p,
      match spec.K.Loop_spec.numerics with
      | None -> []
      | Some n -> [ Note.Numerics_not_applicable n ] )

let js_loop spec inv : (made, Refusal.t) result =
  let* () = js_policy spec.K.Loop_spec.numerics in
  let* p = loop_program spec inv in
  Ok (Language.Js, Loop_js.emit p, [])

let c_loop spec inv : (made, Refusal.t) result =
  let* p = loop_program spec inv in
  let numerics, vector =
    loop_vector Language.C spec.K.Loop_spec.numerics Loop_target.neon128
  in
  match
    (Err.payload (Loop_c.kernel ?vector ~numerics ~name:kernel_name p)
      : (Loop_c.t, Loop_c.error) result)
  with
  | Error e -> Error (Refusal.Loop_c e)
  | Ok k -> Ok (Language.C, c_text k, vector_notes numerics vector @ c_notes k)

let wat_loop spec inv : (made, Refusal.t) result =
  let* p = loop_program spec inv in
  let numerics, vector =
    loop_vector Language.Wat spec.K.Loop_spec.numerics Loop_target.wasm128
  in
  match
    (Err.payload (Loop_wasm.lower ?vector ~numerics p)
      : (Loop_wasm.t, Loop_wasm.error) result)
  with
  | Error e -> Error (Refusal.Loop_wasm e)
  | Ok m ->
      Ok
        ( Language.Wat,
          Wasm_wat.to_string m.Loop_wasm.module_,
          vector_notes numerics vector
          @ [ Note.Working_precision m.Loop_wasm.precision ] )

(* ---- structured SSA ---- *)

let ssa_numerics : Loop_numerics.t -> Ssa_ir.Ssa_numerics.t = function
  | Loop_numerics.Fp32_contracted -> Ssa_ir.Ssa_numerics.Fp32_contracted
  | Loop_numerics.Fp32_ordered -> Ssa_ir.Ssa_numerics.Fp32_ordered
  | Loop_numerics.Reference_f64 -> Ssa_ir.Ssa_numerics.Reference_f64
  | Loop_numerics.Simd_fp32_ordered -> Ssa_ir.Ssa_numerics.Simd_fp32_ordered
  | Loop_numerics.Simd_fp32_relaxed -> Ssa_ir.Ssa_numerics.Simd_fp32_relaxed

let loop_precision : Ssa_ir.Ssa_numerics.Precision.t -> _ = function
  | Ssa_ir.Ssa_numerics.Precision.F32 -> Loop_numerics.Precision.F32
  | Ssa_ir.Ssa_numerics.Precision.F64 -> Loop_numerics.Precision.F64

let ssa_frontend : K.Frontend.t -> Ssa_backends.Frontend.t = function
  | K.Frontend.Direct -> Ssa_backends.Frontend.Direct
  | K.Frontend.Structured -> Ssa_backends.Frontend.Structured

(* The pipeline a family's SSA route runs, and the notes that say what it was
   planned for. A policy other than the reference one plans for the family's
   vector unit, as the drivers do; the reference policy is scalar. *)
let ssa_pipeline family (spec : K.Ssa_spec.t) =
  match spec.K.Ssa_spec.pipeline with
  | K.Pipeline.Exact -> Ok (Ssa_backends.Pipeline.Exact, [])
  | K.Pipeline.Representation -> Ok (Ssa_backends.Pipeline.Representation, [])
  | K.Pipeline.Planned ->
      let* () =
        match family with
        | Language.Js -> js_policy spec.K.Ssa_spec.numerics
        | Language.C | Language.Coral_estimate | Language.Coral_plan
        | Language.Gnu_asm _ | Language.Loop | Language.Machine_ir
        | Language.Ssa | Language.Wat ->
            Ok ()
      in
      let n = numerics_of family spec.K.Ssa_spec.numerics in
      let target =
        if n = Loop_numerics.Reference_f64 then Ssa_ir.Ssa_target.scalar
        else
          match family with
          | Language.Wat -> Ssa_ir.Ssa_target.wasm128
          | Language.C | Language.Coral_estimate | Language.Coral_plan
          | Language.Gnu_asm _ | Language.Js | Language.Loop
          | Language.Machine_ir | Language.Ssa ->
              Ssa_ir.Ssa_target.neon128
      in
      Ok
        ( Ssa_backends.Pipeline.Planned { numerics = ssa_numerics n; target },
          [ Note.Numerics n; Note.Ssa_target target ] )

(* The lowering's own typed refusal comes first: everything after it in
   [Ssa_backends] is a string. *)
let ssa_lowered (spec : K.Ssa_spec.t) inv =
  Result.map_error
    (fun r -> Refusal.Ssa_lowering r)
    (Ssa_backends.lowered ~frontend:(ssa_frontend spec.K.Ssa_spec.frontend) inv)

let ssa_string r = Result.map_error (fun s -> Refusal.Ssa_backends s) r

let ssa_family (spec : K.Ssa_spec.t) inv : (made, Refusal.t) result =
  let frontend = ssa_frontend spec.K.Ssa_spec.frontend in
  let* lowered = ssa_lowered spec inv in
  let* pipeline, notes = ssa_pipeline Language.Ssa spec in
  match pipeline with
  | Ssa_backends.Pipeline.Representation ->
      Ok (Language.Ssa, Ssa_ir.Ssa_pp.to_string lowered, notes)
  | Ssa_backends.Pipeline.Exact ->
      let* p = ssa_string (Ssa_backends.program ~frontend pipeline inv) in
      Ok (Language.Ssa, Ssa_ir.Ssa_pp.to_string p, notes)
  | Ssa_backends.Pipeline.Planned { numerics; target } ->
      let* plan =
        ssa_string (Ssa_backends.plan ~frontend ~numerics ~target inv)
      in
      let module P = Ssa_ir.Ssa_plan in
      Ok
        ( Language.Ssa,
          Ssa_ir.Ssa_pp.to_string plan.P.program,
          notes
          @ List.concat
              [
                [ Note.Working_precision (loop_precision plan.P.precision) ];
                (match plan.P.refusal with
                | None -> []
                | Some r -> [ Note.Ssa_fp32_refusal r ]);
                [ Note.Vectorization plan.P.vectorized ];
                (if plan.P.blocked > 0 then [ Note.Row_blocking ] else []);
                (if plan.P.contracted > 0 then [ Note.Fused_multiply_adds ]
                 else []);
              ] )

let c_ssa spec inv : (made, Refusal.t) result =
  let* _ = ssa_lowered spec inv in
  let* pipeline, notes = ssa_pipeline Language.C spec in
  let* k =
    ssa_string
      (Ssa_backends.c_via
         ~frontend:(ssa_frontend spec.K.Ssa_spec.frontend)
         ~pipeline ~name:kernel_name inv)
  in
  Ok (Language.C, c_text k, notes @ c_notes k)

let js_ssa spec inv : (made, Refusal.t) result =
  let* _ = ssa_lowered spec inv in
  let* pipeline, notes = ssa_pipeline Language.Js spec in
  let* ast =
    ssa_string
      (Ssa_backends.js_via
         ~frontend:(ssa_frontend spec.K.Ssa_spec.frontend)
         ~pipeline inv)
  in
  Ok (Language.Js, Js_print.script ast, notes)

(* A kernel becomes a module the way the bundle builders link one: through the
   module layout, which fixes where its constant tables live. *)
let wat_ssa spec inv : (made, Refusal.t) result =
  let* _ = ssa_lowered spec inv in
  let* pipeline, notes = ssa_pipeline Language.Wat spec in
  let numerics = numerics_of Language.Wat spec.K.Ssa_spec.numerics in
  let linked =
    Err.payload
      (Loop_wasm_module.lower_with ~numerics (fun ~mark_base:_ ~table_alloc ->
           match
             Ssa_backends.wasm_via
               ~frontend:(ssa_frontend spec.K.Ssa_spec.frontend)
               ~pipeline ~table_alloc inv
           with
           | Ok k -> Err.return k
           | Error s -> Err.fail s))
  in
  match linked with
  | Error s -> Error (Refusal.Ssa_backends s)
  | Ok m ->
      Ok
        ( Language.Wat,
          Wasm_wat.to_string m.Loop_wasm_module.module_,
          notes @ [ Note.Working_precision m.Loop_wasm_module.precision ] )

(* ---- Machine IR ---- *)

(* What every target selects: binary32, ordered, scalar. The key names no policy
   (a target admits only this plan, and a vector form it lacks is a refusal), so
   the notes say what ran. *)
let mir_pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

(* The invocation's position in the schedule is not known to a per-node view; a
   refusal reports 0. *)
let mir_family (spec : K.Mir_spec.t) inv : (made, Refusal.t) result =
  let module M = Machine_model.Mir_model in
  match M.lower_one ~pipeline:mir_pipeline ~invocation:0l inv with
  | Error r -> Error (Refusal.Mir_lowering r)
  | Ok lowered -> (
      match
        M.Stage_text.of_lowered spec.K.Mir_spec.target spec.K.Mir_spec.stage
          lowered
      with
      | Error r -> Error (Refusal.Mir_stage r)
      | Ok { M.Stage_text.text; note } ->
          Ok
            ( Language.Machine_ir,
              text,
              [
                Note.Numerics Loop_numerics.Fp32_ordered;
                Note.Ssa_target Ssa_ir.Ssa_target.scalar;
              ]
              @ Option.fold ~none:[]
                  ~some:(fun s -> [ Note.Mir_summary s ])
                  note ))

(* ---- assembly ---- *)

let asm_family target inv : (made, Refusal.t) result =
  match Lowering_asm.of_invocation target inv with
  | Error r -> Error r
  | Ok (text, notes) -> Ok (Language.Gnu_asm target, text, notes)

(* ---- dispatch ---- *)

let route ~loop ~ssa = function
  | K.Route.Loop spec -> loop spec
  | K.Route.Ssa spec -> ssa spec

let make ~granted (inv : Loop_bundle.invocation) variant :
    (made, Refusal.t) result =
  if granted inv.Loop_bundle.node then
    Error (Refusal.Granted inv.Loop_bundle.node)
  else
    match variant with
    | K.Asm target -> asm_family target inv
    | K.C r -> route ~loop:(fun s -> c_loop s inv) ~ssa:(fun s -> c_ssa s inv) r
    | K.Js r ->
        route ~loop:(fun s -> js_loop s inv) ~ssa:(fun s -> js_ssa s inv) r
    | K.Loop spec -> loop_family spec inv
    | K.Machine_ir spec -> mir_family spec inv
    | K.Ssa spec -> ssa_family spec inv
    | K.Wat r ->
        route ~loop:(fun s -> wat_loop s inv) ~ssa:(fun s -> wat_ssa s inv) r
    | K.Coral v -> Lowering_coral.of_invocation v inv

let artifact ?dialect node variant result =
  let key = K.make ?dialect node variant in
  match result with
  | Error r -> { Artifact.key; body = Artifact.Refused r; notes = [] }
  | Ok ((language, text, notes) : made) ->
      {
        Artifact.key;
        body = Artifact.Text { language; text; truncated = false };
        notes;
      }

let of_invocation ?(dialect = K.Dialect.Native) ~granted
    (inv : Loop_bundle.invocation) variant =
  let family = K.family variant in
  artifact ~dialect inv.Loop_bundle.node variant
    (if K.reaches dialect family then make ~granted inv variant
     else Error (Refusal.Unsupported_dialect { dialect; family }))

let of_node ?(dialect = K.Dialect.Native) ~granted (b : Loop_bundle.t) node
    variant =
  match
    List.find_opt
      (fun (inv : Loop_bundle.invocation) ->
        Graph_ir.Node_id.equal inv.Loop_bundle.node node)
      b.Loop_bundle.invocations
  with
  | Some inv -> of_invocation ~dialect ~granted inv variant
  | None -> artifact ~dialect node variant (Error (Refusal.Unknown_node node))
