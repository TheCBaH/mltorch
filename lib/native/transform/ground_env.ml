(* The per-graph lookup environment for [Ground_eval]; its public face is
   [Ground_eval.Env] (see ground_eval.mli). Split out of ground_eval.ml under
   the tracked file-size ceiling (scripts/check-file-size.sh). The lookups
   beyond that face ([const_of], [constant_of], [stage_of], [stage_of_id], ...)
   are what grounding itself reads. *)

open Graph_ir
module Origin = Ground_expr.Origin

(* EVERYTHING is keyed by this graph's raw id, and the only thing [side] does
   is tag the cells grounding emits so the two graphs' [t2]s are distinguish-
   able. That split is the point: a correspondence variable is a claim about
   the map, and every question this module answers — which stage produces an
   edge, what format it is stored in, whether a payload is bound — is a
   question about one graph that the claim must not be able to reach.

   It replaces an [origin : Tensor_id.t -> Origin.t] that could rewrite an id
   into a variable before those lookups ran, and a [var_edge] map that existed
   only to undo that. *)
type t = {
  constant_ids : Tensor_id.Set.t;
  constants : Tensor.packed Tensor_id.Map.t;
  constant_store : Constant_store.t option;
  consts : float Tensor_id.Map.t;
  fmts : Payload.packed_fmt Tensor_id.Map.t;
  inputs : Tensor_id.Set.t;
  shapes : Vec6.shape Tensor_id.Map.t;
  side : [ `Dst | `Src ];
  stages : Stage_program.Stage.t Tensor_id.Map.t;
  i64_stages : Stage_program.Stage_i64.t Tensor_id.Map.t;
}

let of_program ?(constants = Tensor_id.Map.empty) ?constant_store
    (p : Stage_program.t) ~side =
  let add_sig (sg : Tensor_sig.t) (fmts, shapes) =
    let id = sg.Tensor_sig.id in
    ( Tensor_id.Map.add id sg.Tensor_sig.fmt fmts,
      Tensor_id.Map.add id sg.Tensor_sig.shape shapes )
  in
  let seed = (Tensor_id.Map.empty, Tensor_id.Map.empty) in
  let fmts, shapes =
    List.fold_left
      (fun acc (_, sg) -> add_sig sg acc)
      seed p.Stage_program.inputs
  in
  let fmts, shapes =
    List.fold_left
      (fun acc (sg, _) -> add_sig sg acc)
      (fmts, shapes) p.Stage_program.consts
  in
  let fmts, shapes =
    List.fold_left
      (fun acc (st : Stage_program.Stage.t) -> add_sig st.sg acc)
      (fmts, shapes) p.Stage_program.stages
  in
  let (fmts, shapes), i64_stages =
    List.fold_left
      (fun (acc, m) (st : Stage_program.Stage_i64.t) ->
        (add_sig st.sg acc, Tensor_id.Map.add st.Stage_program.Stage_i64.id st m))
      ((fmts, shapes), Tensor_id.Map.empty)
      p.Stage_program.stages_i64
  in
  let consts =
    List.fold_left
      (fun acc ((sg : Tensor_sig.t), v) ->
        Tensor_id.Map.add sg.Tensor_sig.id v acc)
      Tensor_id.Map.empty p.Stage_program.consts
  in
  let stages =
    List.fold_left
      (fun acc (st : Stage_program.Stage.t) ->
        Tensor_id.Map.add st.Stage_program.Stage.id st acc)
      Tensor_id.Map.empty p.Stage_program.stages
  in
  let inputs =
    List.fold_left
      (fun acc ((id : Tensor_id.t), _) -> Tensor_id.Set.add id acc)
      Tensor_id.Set.empty p.Stage_program.inputs
  in
  (* Which of this graph's inputs are MODEL CONSTANTS. Membership in
     [Stage_program.inputs] and the kind, both: [input_kinds] keys inputs only
     and is sparse, so [None] means [Input] and an internal edge is absent
     from it entirely. *)
  let constant_ids =
    List.fold_left
      (fun acc ((id : Tensor_id.t), _) ->
        match Tensor_id.Map.find_opt id p.Stage_program.input_kinds with
        | None -> acc
        | Some Input.Constant -> Tensor_id.Set.add id acc
        | Some Input.Input -> acc)
      Tensor_id.Set.empty p.Stage_program.inputs
  in
  {
    constant_ids;
    constants;
    constant_store;
    consts;
    fmts;
    inputs;
    shapes;
    side;
    stages;
    i64_stages;
  }

(* The cell this graph's [id] reads as. Side-qualified, always: turning it
   into a correspondence variable is [Ground_expr.project]'s business, and
   happens on a copy at comparison time. *)
let origin t id =
  match t.side with `Dst -> Origin.Dst id | `Src -> Origin.Src id

(* The edge of THIS graph an origin denotes, if any. A [Boundary] names none —
   it stands for a value both graphs hold, not for one edge — so every lookup
   below answers conservatively for one, which is what keeps a projected term
   from being normalised or expanded by accident. *)
let edge_of _t (o : Origin.t) = Origin.edge o

let shape_of t o =
  Option.bind (edge_of t o) (fun id -> Tensor_id.Map.find_opt id t.shapes)

(* F32/F16/BF16 decode to a value already representable in f32 (they carry no
   more mantissa). I32/I64 go through [Int32.to_float]/[Int64.to_float], which
   exceeds f32's exact range above 2^24; I8/I16 go through
   [Quant.dequantize], a scale multiply whose product need not be
   f32-representable either. See [Payload.get_float]. *)
let fmt_is_f32_exact (Payload.Fmt fmt) =
  match fmt with
  | Payload.Bool | Payload.F32 | Payload.F16 | Payload.BF16 -> true
  | Payload.F64 | Payload.I8 | Payload.I16 | Payload.I32 | Payload.I64 -> false

let stored_f32 t (cell : Ground_expr.Cell.t) =
  let fmt =
    match cell.Ground_expr.Cell.origin with
    | Ground_expr.Origin.Capture capture ->
        Option.bind t.constant_store (fun store ->
            Const_ssa_symbolic.captured_fmt store capture)
    | Ground_expr.Origin.Boundary _ | Ground_expr.Origin.Dst _
    | Ground_expr.Origin.Src _ ->
        Option.bind (edge_of t cell.Ground_expr.Cell.origin) (fun id ->
            Tensor_id.Map.find_opt id t.fmts)
  in
  Option.fold ~none:false ~some:fmt_is_f32_exact fmt

let const_of t id = Tensor_id.Map.find_opt id t.consts
let constant_of t id = Tensor_id.Map.find_opt id t.constants

(* USER data, so both conditions: a graph input of this program AND not a
   model constant. [input_kinds] is sparse and keys inputs only, so [None]
   means [Input] — and an internal edge, absent from it entirely, would answer
   yes to the kind alone. See the `.ai/` native transform verify doc §9a. *)
let is_user_input t id =
  Tensor_id.Set.mem id t.inputs && not (Tensor_id.Set.mem id t.constant_ids)

(* A model constant whose payload was not supplied. Such a cell is free only
   because nothing bound it, not because anything may vary over it, so the
   value tiers must not run against one: a probe would separate two constants
   that may well hold the same bytes. *)
let unbound_constant t (cell : Ground_expr.Cell.t) =
  edge_of t cell.Ground_expr.Cell.origin
  |> Option.fold ~none:false ~some:(fun id ->
      Tensor_id.Set.mem id t.constant_ids
      && Option.is_none (const_of t id)
      && Option.is_none (constant_of t id))

let stage_of t o =
  Option.bind (Origin.edge o) (fun id -> Tensor_id.Map.find_opt id t.stages)

let stage_of_id t id = Tensor_id.Map.find_opt id t.stages
