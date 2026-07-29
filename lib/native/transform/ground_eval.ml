(* See ground_eval.mli. *)

open Graph_ir
module Origin = Ground_expr.Origin

module Env = struct
  (* Metadata is keyed by THIS SIDE's raw id, and only a cell's identity uses the
     origin. The split matters because [origin] is dynamic — an edge becomes
     [Shared] once its own cluster is proved (see [Map_verify]) — so keying the
     maps by origin would strand every entry the moment its key changed.

     [var_edge] closes the one gap that leaves: an [Input v] cell names no edge,
     because the two sides' input ids differ, so each side records which of its
     own edges the variable stands for. *)
  type t = {
    constants : Tensor.packed Tensor_id.Map.t;
    consts : float Tensor_id.Map.t;
    fmts : Payload.packed_fmt Tensor_id.Map.t;
    shapes : Vec6.shape Tensor_id.Map.t;
    origin : Tensor_id.t -> Origin.t;
    stages : Stage_program.Stage.t Tensor_id.Map.t;
    var_edge : Tensor_id.t Input_var.Map.t;
  }

  let of_program ?(constants = Tensor_id.Map.empty) (p : Stage_program.t)
      ~origin =
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
    (* Which of this graph's edges each sigma variable stands for. Read off the
       inputs, whose classification is fixed before any cluster is checked. *)
    let var_edge =
      List.fold_left
        (fun acc (_, (sg : Tensor_sig.t)) ->
          match origin sg.Tensor_sig.id with
          | Origin.Input v -> Input_var.Map.add v sg.Tensor_sig.id acc
          | _ -> acc)
        Input_var.Map.empty p.Stage_program.inputs
    in
    { constants; consts; fmts; origin; shapes; stages; var_edge }

  (* The edge of THIS graph an origin denotes, if any. *)
  let edge_of t (o : Origin.t) =
    match o with
    | Origin.Dst id | Origin.Shared id | Origin.Src id -> Some id
    | Origin.Input v -> Input_var.Map.find_opt v t.var_edge

  let shape_of t o =
    Option.bind (edge_of t o) (fun id -> Tensor_id.Map.find_opt id t.shapes)

  (* F32/F16/BF16 decode to a value already representable in f32 (they carry no
     more mantissa). I32/I64 go through [Int32.to_float]/[Int64.to_float], which
     exceeds f32's exact range above 2^24; I8/I16 go through
     [Quant.dequantize], a scale multiply whose product need not be
     f32-representable either. See [Payload.get_float]. *)
  let fmt_is_f32_exact (Payload.Fmt fmt) =
    match fmt with
    | Payload.F32 | Payload.F16 | Payload.BF16 -> true
    | Payload.I8 | Payload.I16 | Payload.I32 | Payload.I64 -> false

  let stored_f32 t (cell : Ground_expr.Cell.t) =
    match
      Option.bind (edge_of t cell.Ground_expr.Cell.origin) (fun id ->
          Tensor_id.Map.find_opt id t.fmts)
    with
    | Some fmt -> fmt_is_f32_exact fmt
    | None -> false

  let const_of t id = Tensor_id.Map.find_opt id t.consts
  let constant_of t id = Tensor_id.Map.find_opt id t.constants
  let origin t id = t.origin id

  (* An [Input v] cell is a graph input on this side and so has no stage, which
     is why this goes through [Origin.edge] rather than [edge_of]. *)
  let stage_of t o =
    match Origin.edge o with
    | Some id -> Tensor_id.Map.find_opt id t.stages
    | None -> None
end

type error = [ `Unknown_edge of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unknown_edge id -> Fmt.pf fmt "unknown edge %a" Tensor_id.pp id

(* ---- grounding one stage body -------------------------------------------- *)

(* [Load]s become leaves through [leaf]; every index is evaluated at [coord]
   (and at the enclosing reduction variables), which is what removes the
   binders. *)
let rec ground ~env ~coord ~rvars (e : Expr.t) : Ground_expr.t =
  let recur = ground ~env ~coord ~rvars in
  let index i = Expr.eval_index_expr ~coord ~rvars i in
  match e with
  | Expr.Const x -> Ground_expr.Const x
  | Expr.Binary (op, a, b) -> Ground_expr.Binary (op, recur a, recur b)
  | Expr.Unary (op, x) -> Ground_expr.Unary (op, recur x)
  | Expr.Value_of_index i -> Ground_expr.Const (float_of_int (index i))
  | Expr.Select (c, a, b) -> (
      match c with
      (* An index comparison is decided by the coordinate, so the [Select]
         collapses outright rather than surviving as a guard. *)
      | Expr.Index_eq (x, y) ->
          if Int.equal (index x) (index y) then recur a else recur b
      | Expr.Cmp (Expr.Lt, x, y) ->
          Ground_expr.Select
            (Ground_expr.Lt (recur x, recur y), recur a, recur b))
  | Expr.Load (sg, idx) -> leaf ~env sg (fun a -> index (Vec6.get idx a))
  | Expr.Max_pool { input; kernel; stride; pad; out; result } ->
      max_pool ~env ~coord ~rvars ~input ~kernel ~stride ~pad ~out ~result
  | Expr.Reduce { kind; var; lo; hi; body } ->
      let lo = index lo and hi = index hi in
      let combine, seed =
        match kind with
        | Expr.Sum ->
            ( (fun a b -> Ground_expr.Binary (Expr.Add, a, b)),
              Ground_expr.Const 0. )
        | Expr.Max_reduce ->
            ( (fun a b -> Ground_expr.Max (Max_op.Float_max, a, b)),
              Ground_expr.Const neg_infinity )
      in
      (* Same left fold, same seed and same order as [Expr.eval]'s arm — the
         ground form has to reproduce the engine's association, not merely its
         value. *)
      let rec fold i acc =
        if i >= hi then acc
        else
          fold (i + 1)
            (combine acc (ground ~env ~coord ~rvars:((var, i) :: rvars) body))
      in
      fold lo seed

(* A [Load] of a synthetic constant fill is that constant; a [Load] of a bound
   model constant is its stored element, read exactly the way [Expr.eval] reads
   it; anything else is a free cell. *)
and leaf ~env (sg : Tensor_sig.t) at_axis : Ground_expr.t =
  let id = sg.Tensor_sig.id in
  match (Env.const_of env id, Env.constant_of env id) with
  | Some v, _ -> Ground_expr.Const v
  | None, Some payload -> Ground_expr.Const (Tensor.read_at_raw payload at_axis)
  | None, None ->
      Ground_expr.Cell
        {
          Ground_expr.Cell.origin = Env.origin env id;
          coord =
            Vec6.coord ~n:(at_axis Axis.N) ~t:(at_axis Axis.T)
              ~d:(at_axis Axis.D) ~h:(at_axis Axis.H) ~w:(at_axis Axis.W)
              ~c:(at_axis Axis.C);
        }

(* The window is concrete, so the stencil expands into the same paired fold
   [Direct]/[Expr.eval] run: one predicate advancing value and index together.
   The value accumulator is a binary [Max] node, so it is mentioned once and
   the fold stays linear in the window; the index accumulator has to name the
   running best inside its guard, which is why it is the larger of the two. *)
and max_pool ~env ~coord ~rvars ~input ~kernel ~stride ~pad ~out ~result =
  let index i = Expr.eval_index_expr ~coord ~rvars i in
  let kh, kw = ((kernel.Op_config.Hw.h :> int), (kernel.Op_config.Hw.w :> int))
  and sh, sw = ((stride.Op_config.Hw.h :> int), (stride.Op_config.Hw.w :> int))
  and ph, pw = ((pad.Op_config.Hw.h :> int), (pad.Op_config.Hw.w :> int)) in
  let h = (Vec6.get input.Tensor_sig.shape Axis.H :> int)
  and w = (Vec6.get input.Tensor_sig.shape Axis.W :> int) in
  let out_h = index (Vec6.get out Axis.H)
  and out_w = index (Vec6.get out Axis.W) in
  let hlo = Stdlib.max 0 ((out_h * sh) - ph)
  and hhi = Stdlib.min h ((out_h * sh) - ph + kh) in
  let wlo = Stdlib.max 0 ((out_w * sw) - pw)
  and whi = Stdlib.min w ((out_w * sw) - pw + kw) in
  let read ih iw =
    leaf ~env input (fun a ->
        if a = Axis.H then ih
        else if a = Axis.W then iw
        else index (Vec6.get out a))
  in
  let rec fold_w ih iw (best, best_index) =
    if iw >= whi then fold_h (ih + 1) (best, best_index)
    else
      let v = read ih iw in
      fold_w ih (iw + 1)
        ( Ground_expr.Max (Max_op.Pool_max, best, v),
          Ground_expr.Select
            ( Ground_expr.Pool_better { best; value = v },
              Ground_expr.Const (float_of_int ((ih * w) + iw)),
              best_index ) )
  and fold_h ih acc = if ih >= hhi then acc else fold_w ih wlo acc in
  let best, best_index =
    fold_h hlo (Ground_expr.Const neg_infinity, Ground_expr.Const 0.)
  in
  match result with
  | Expr.Max_pool.Value -> best
  | Expr.Max_pool.Index -> best_index

(* ---- the interface -------------------------------------------------------- *)

let body_at env (st : Stage_program.Stage.t) coord =
  ground ~env
    ~coord:(Schedule.coord_index coord)
    ~rvars:[] st.Stage_program.Stage.body

let at env id coord =
  (* [id] names an edge of THIS graph, so the stage lookup goes through the
     origin the same way expansion does. *)
  match Env.stage_of env (Env.origin env id) with
  | Some st -> Core.return (Ground_expr.Round (body_at env st coord))
  | None -> (
      (* An input edge can itself be a bound constant — [fold_const]'s whole
         output is one — so the same binding [leaf] applies inside a body has to
         apply here too, not just to [Load]s. *)
      match (Env.const_of env id, Env.constant_of env id) with
      | Some v, _ -> Core.return (Ground_expr.Const v)
      | None, Some payload ->
          Core.return
            (Ground_expr.Const
               (Tensor.read_at_raw payload (fun a ->
                    Dim.to_int (Vec6.get coord a))))
      | None, None ->
          let origin = Env.origin env id in
          if Option.is_none (Env.shape_of env origin) then
            Core.fail (`Unknown_edge id)
          else Core.return (Ground_expr.Cell { Ground_expr.Cell.origin; coord })
      )

(* [budget] bounds ONE round, and has to: a single substitution step is
   quadratic where a conv feeds a conv, so a term can reach tens of millions of
   nodes before anyone gets to measure it. Measuring afterwards made verifying a
   real model cost 25x the transform it was checking.

   Running out mid-round leaves the remaining cells unexpanded, which is sound
   rather than approximate: an unexpanded cell keeps [expandable] true, so the
   driver reports a budget verdict, and no probe may run against a frontier that
   never reached the inputs. The node count is threaded, not counted in a ref. *)
let expand ~budget env (e : Ground_expr.t) : Ground_expr.t =
  let rec go n e =
    if n >= budget then (n, e)
    else
      match e with
      | Ground_expr.Cell c -> (
          match Env.stage_of env c.Ground_expr.Cell.origin with
          | Some st ->
              let body =
                Ground_expr.Round (body_at env st c.Ground_expr.Cell.coord)
              in
              (n + Ground_expr.size body, body)
          | None -> (n + 1, e))
      | Ground_expr.Const _ -> (n + 1, e)
      | Ground_expr.Binary (op, a, b) ->
          let n, a = go (n + 1) a in
          let n, b = go n b in
          (n, Ground_expr.Binary (op, a, b))
      | Ground_expr.Max (op, a, b) ->
          let n, a = go (n + 1) a in
          let n, b = go n b in
          (n, Ground_expr.Max (op, a, b))
      | Ground_expr.Round x ->
          let n, x = go (n + 1) x in
          (n, Ground_expr.Round x)
      | Ground_expr.Unary (op, x) ->
          let n, x = go (n + 1) x in
          (n, Ground_expr.Unary (op, x))
      | Ground_expr.Select (g, a, b) ->
          let n, g =
            match g with
            | Ground_expr.Lt (x, y) ->
                let n, x = go (n + 1) x in
                let n, y = go n y in
                (n, Ground_expr.Lt (x, y))
            | Ground_expr.Pool_better { best; value } ->
                let n, best = go (n + 1) best in
                let n, value = go n value in
                (n, Ground_expr.Pool_better { best; value })
          in
          let n, a = go n a in
          let n, b = go n b in
          (n, Ground_expr.Select (g, a, b))
  in
  snd (go 0 e)

let expandable env e =
  Ground_expr.Cell.Set.exists
    (fun (c : Ground_expr.Cell.t) ->
      Option.is_some (Env.stage_of env c.Ground_expr.Cell.origin))
    (Ground_expr.cells e)

(* [Set.find_first_opt] is NOT usable here: it requires a monotone predicate and
   silently misbehaves on an arbitrary one. *)
let out_of_bounds env e =
  List.find_opt
    (fun (c : Ground_expr.Cell.t) ->
      match Env.shape_of env c.Ground_expr.Cell.origin with
      | Some shape -> not (Vec6.in_bounds shape c.Ground_expr.Cell.coord)
      | None -> false)
    (Ground_expr.Cell.Set.elements (Ground_expr.cells e))
