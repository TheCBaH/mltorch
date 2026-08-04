(* See ground_eval.mli. *)

open Graph_ir
module Origin = Ground_expr.Origin

module Env = struct
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
    consts : float Tensor_id.Map.t;
    fmts : Payload.packed_fmt Tensor_id.Map.t;
    inputs : Tensor_id.Set.t;
    shapes : Vec6.shape Tensor_id.Map.t;
    side : [ `Dst | `Src ];
    stages : Stage_program.Stage.t Tensor_id.Map.t;
  }

  let of_program ?(constants = Tensor_id.Map.empty) (p : Stage_program.t) ~side
      =
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
          | Some Input.Constant -> Tensor_id.Set.add id acc
          | None | Some Input.Input -> acc)
        Tensor_id.Set.empty p.Stage_program.inputs
    in
    { constant_ids; constants; consts; fmts; inputs; shapes; side; stages }

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
    | Payload.F32 | Payload.F16 | Payload.BF16 -> true
    | Payload.I8 | Payload.I16 | Payload.I32 | Payload.I64 -> false

  let stored_f32 t (cell : Ground_expr.Cell.t) =
    Option.bind (edge_of t cell.Ground_expr.Cell.origin) (fun id ->
        Tensor_id.Map.find_opt id t.fmts)
    |> Option.fold ~none:false ~some:fmt_is_f32_exact

  let const_of t id = Tensor_id.Map.find_opt id t.consts
  let constant_of t id = Tensor_id.Map.find_opt id t.constants

  (* USER data, so both conditions: a graph input of this program AND not a
     model constant. [input_kinds] is sparse and keys inputs only, so [None]
     means [Input] — and an internal edge, absent from it entirely, would answer
     yes to the kind alone. See .ai/native_transform_verify.md §9a. *)
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
end

type error = [ `Unknown_edge of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unknown_edge id -> Fmt.pf fmt "unknown edge %a" Tensor_id.pp id

(* ---- grounding one stage body -------------------------------------------- *)

(* [Load]s become leaves through [leaf]; every index is evaluated at [coord]
   (and at the enclosing reduction variables), which is what removes the
   binders. *)
let rec ground ~env ~coord ~rvars (e : Symbolic_expr.t) : Ground_expr.t =
  let recur = ground ~env ~coord ~rvars in
  let index i = Symbolic_expr.eval_index_expr ~coord ~rvars i in
  match e with
  | Symbolic_expr.Const x -> Ground_expr.Const x
  | Symbolic_expr.Binary (op, a, b) -> Ground_expr.Binary (op, recur a, recur b)
  | Symbolic_expr.Unary (op, x) -> Ground_expr.Unary (op, recur x)
  | Symbolic_expr.Value_of_index i -> Ground_expr.Const (float_of_int (index i))
  | Symbolic_expr.Select (c, a, b) -> (
      match c with
      (* An index comparison is decided by the coordinate, so the [Select]
         collapses outright rather than surviving as a guard. *)
      | Symbolic_expr.Index_eq (x, y) ->
          if Int.equal (index x) (index y) then recur a else recur b
      | Symbolic_expr.Cmp (Symbolic_expr.Lt, x, y) ->
          Ground_expr.Select
            (Ground_expr.Lt (recur x, recur y), recur a, recur b))
  | Symbolic_expr.Load (sg, idx) ->
      leaf ~env sg (fun a -> index (Vec6.get idx a))
  | Symbolic_expr.Max_pool { input; kernel; stride; pad; out; result } ->
      max_pool ~env ~coord ~rvars ~input ~kernel ~stride ~pad ~out ~result
  | Symbolic_expr.Reduce { kind; var; lo; hi; body } ->
      let lo = index lo and hi = index hi in
      let combine, seed =
        match kind with
        | Symbolic_expr.Sum ->
            ( (fun a b -> Ground_expr.Binary (Symbolic_expr.Add, a, b)),
              Ground_expr.Const 0. )
        | Symbolic_expr.Max_reduce ->
            ( (fun a b -> Ground_expr.Max (Max_op.Float_max, a, b)),
              Ground_expr.Const neg_infinity )
      in
      (* Same left fold, same seed and same order as [Symbolic_expr.eval]'s arm — the
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
   model constant is its stored element, read exactly the way [Symbolic_expr.eval] reads
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
   [Direct]/[Symbolic_expr.eval] run: one predicate advancing value and index together.
   The value accumulator is a binary [Max] node, so it is mentioned once and
   the fold stays linear in the window; the index accumulator has to name the
   running best inside its guard, which is why it is the larger of the two. *)
and max_pool ~env ~coord ~rvars ~input ~kernel ~stride ~pad ~out ~result =
  let index i = Symbolic_expr.eval_index_expr ~coord ~rvars i in
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
  | Symbolic_expr.Max_pool.Value -> best
  | Symbolic_expr.Max_pool.Index -> best_index

(* ---- the interface -------------------------------------------------------- *)

let body_at env (st : Stage_program.Stage.t) coord =
  ground ~env
    ~coord:(Schedule.coord_index coord)
    ~rvars:[] st.Stage_program.Stage.body

let at env id coord =
  (* [id] names an edge of THIS graph, so the stage is looked up by that raw id
     DIRECTLY. A root is never replaced by a correspondence variable, whatever
     cluster it is in: doing so would let the cluster under test discharge
     itself by naming both its sides the same thing before either definition was
     looked at. Expanding the root is what puts a real definition on each side;
     projection then applies to what that reads. *)
  match Env.stage_of_id env id with
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
   never reached the inputs. The node count is threaded, not counted in a ref.

   [boundary] stops it at the LOCAL frontier: a cell whose cluster gives it a
   variable is a free variable of this obligation, so expanding through it would
   be re-proving someone else's. That is the difference between a budget
   truncation and a completed frontier, and the two must not be confused —
   which is why [expandable] below asks the same question. *)
let expand ~boundary ~budget env (e : Ground_expr.t) : Ground_expr.t =
  let rec go n e =
    if n >= budget then (n, e)
    else
      match e with
      | Ground_expr.Cell c
        when Option.is_some (boundary c.Ground_expr.Cell.origin) ->
          (n + 1, e)
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

let expandable ~boundary env e =
  Ground_expr.Cell.Set.exists
    (fun (c : Ground_expr.Cell.t) ->
      Option.is_none (boundary c.Ground_expr.Cell.origin)
      && Option.is_some (Env.stage_of env c.Ground_expr.Cell.origin))
    (Ground_expr.cells e)

(* [Set.find_first_opt] is NOT usable here: it requires a monotone predicate and
   silently misbehaves on an arbitrary one. *)
let out_of_bounds env e =
  List.find_opt
    (fun (c : Ground_expr.Cell.t) ->
      Env.shape_of env c.Ground_expr.Cell.origin
      |> Option.fold ~none:false ~some:(fun shape ->
          not (Vec6.in_bounds shape c.Ground_expr.Cell.coord)))
    (Ground_expr.Cell.Set.elements (Ground_expr.cells e))
