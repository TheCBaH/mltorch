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
    constant_store : Constant_store.t option;
    consts : float Tensor_id.Map.t;
    fmts : Payload.packed_fmt Tensor_id.Map.t;
    inputs : Tensor_id.Set.t;
    (* [Stage.pixel_body] specializes and re-checks a stage's whole computation
       and is stage-invariant, so it is worth caching per stage id rather than
       redone at every coordinate [body_at] grounds. Keyed by stage id alone:
       every caller in this module reaches [pixel_body] through [body_at],
       which always passes [Kernel.Limits.default]. *)
    pixel_bodies : Expr.Value.t Tensor_id.Map.t ref;
    shapes : Vec6.shape Tensor_id.Map.t;
    side : [ `Dst | `Src ];
    stages : Stage_program.Stage.t Tensor_id.Map.t;
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
      pixel_bodies = ref Tensor_id.Map.empty;
      shapes;
      side;
      stages;
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
    | Payload.F32 | Payload.F16 | Payload.BF16 -> true
    | Payload.F64 | Payload.I8 | Payload.I16 | Payload.I32 | Payload.I64 ->
        false

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

  let pixel_body t ~max_size ~max_depth (st : Stage_program.Stage.t) =
    match
      Tensor_id.Map.find_opt st.Stage_program.Stage.id !(t.pixel_bodies)
    with
    | Some body -> Err.return body
    | None ->
        let open Err.Syntax in
        let+ body = Stage_program.Stage.pixel_body ~max_size ~max_depth st in
        t.pixel_bodies :=
          Tensor_id.Map.add st.Stage_program.Stage.id body !(t.pixel_bodies);
        body
end

(* [Expr.Eval.error] joins the row: grounding evaluates indices, and checked
   arithmetic can now fail where it previously wrapped.
   [`Data_index_unresolved] is [resolve_data_source]'s own conservative
   catch-all (below): a [Data] source this grounder cannot resolve to an
   exact value. It feeds [map_verify_check.ml]'s existing generic
   [Ground_eval.error -> Unproved] conversion unchanged -- an unresolved
   [Data] source makes a cluster [Unproved], never a build failure. *)
type error =
  [ Expr.Eval.error
  | `Data_index_unresolved
  | `Region of Region_program.error
  | `Unknown_edge of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Expr.Eval.error as e -> Expr.Eval.pp_error fmt e
  | `Data_index_unresolved ->
      Fmt.string fmt
        "Data index source could not be resolved to a directly-bound I64 \
         constant"
  | `Region e -> Region_program.pp_error fmt e
  | `Unknown_edge id -> Fmt.pf fmt "unknown edge %a" Tensor_id.pp id

(* The exact resolver for a [Data] source during grounding: succeeds ONLY for
   a DIRECTLY BOUND constant, with no stage-walking fallback. An earlier draft
   also followed any stage whose body was structurally an identity load,
   reasoning that a value-preserving expression implies a storage-preserving
   read -- it does not: [Stage_program.ground] evaluates every stage through
   [Tensor.materialize], which unconditionally allocates F32, so an
   identity-load stage sitting between an I64 constant and a [Data] read would
   produce an F32-rounded value under real symbolic execution while the
   stage-walk bypassed that boundary and returned the exact original I64
   value instead -- a verification/execution mismatch. Sufficient for
   CSATv2's real occurrence, since the round-6 import-time trace-back already
   makes [Index_tensor]'s [index] operand the original constant directly,
   with no intervening stage at all. Everything else -- a stage in the way,
   a Const-SSA-backed capture, a non-constant edge -- conservatively falls
   through to [`Data_index_unresolved] rather than computing an answer that
   could disagree with real execution. *)
let resolve_data_source (env : Env.t) (id : Tensor_id.t)
    (coord : int Expr.Coord.t) : (int64, [> error ]) Err.t =
  match Env.constant_of env id with
  | Some t ->
      Tensor.read_i64_at6 t (fun a -> Expr.Coord.get coord a)
      |> Err.map_error (fun _ -> `Data_index_unresolved)
  | None -> Err.fail `Data_index_unresolved

(* Index evaluation is result-returning, but [ground] and [expand] are
   recursive rebuilds threading a node budget; converting at every step would
   rewrite them monadically for a failure that cannot occur on a well-formed
   graph. So the recursion escapes through [Err.Escape] and the module's PUBLIC
   entry points convert once -- the same shape [Expr.Eval] uses internally, and
   for the same reason. [throw_error] carries the already-built [Err.Error.t],
   so the backtrace is the one captured where the failure was detected.

   The local wrapper exists only for the row widening: [Escape.or_throw] is
   monomorphic in the payload, and [Error.t]'s covariance coerces a sub-row for
   free where [map_error] would record a [Map] event this never had. *)
let or_throw esc : ('a, [< error ]) Err.t -> 'a = function
  | Ok v -> v
  | Error e -> Err.Escape.throw_error esc (e :> error Err.Error.t)

(* ---- grounding one stage body -------------------------------------------- *)

(* [Load]s become leaves through [leaf]; every index is evaluated at [coord]
   (and at the enclosing reduction variables), which is what removes the
   binders. *)
let rec ground esc ~env ~coord ~rvars (e : Expr.Value.t) : Ground_expr.t =
  let recur = ground esc ~env ~coord ~rvars in
  (* Calls [eval_index] directly (not the public [Expr.Eval.index]), passing
     THIS module's own escape token: both use the identical escape-based
     non-local-exit pattern, so a [Data] failure inside [eval_index] throws
     straight through to this function's caller with no extra conversion.
     [~widen] lifts the ordinary arithmetic/reducer failures ([index_error])
     up to this row; [~resolve_data] is [resolve_data_source], converting the
     [Source.t] it receives back to a [Tensor_id.t] first. *)
  let index : type r. r Expr.Index.t -> int =
   fun i ->
    Expr.Eval.eval_index esc
      ~widen:(fun (e : Expr.Eval.index_error) -> (e :> error))
      ~output:coord
      ~reducers:(fun v ->
        List.find_map
          (fun (w, n) -> if Expr.Reduce_var.equal v w then Some n else None)
          rvars)
      ~resolve_data:(fun src coord ->
        resolve_data_source env (Expr_bridge.id_of_source src) coord)
      i
  in
  match e with
  | Expr.Value.Const x -> Ground_expr.Const x
  | Expr.Value.Local v -> Err.Escape.throw esc (`Unbound_local v)
  | Expr.Value.Local_at (v, _) -> Err.Escape.throw esc (`Unbound_local v)
  | Expr.Value.Binary (op, a, b) -> Ground_expr.Binary (op, recur a, recur b)
  | Expr.Value.Unary (op, x) -> Ground_expr.Unary (op, recur x)
  | Expr.Value.Value_of_index i ->
      (* Through the shared conversion, not [float_of_int]: this is the second
         interpreter, and a helper returning [int] does not make the conversion
         that follows it exact. *)
      Ground_expr.Const (or_throw esc (Expr.Eval.float_of_index (index i)))
  | Expr.Value.Round_f32 x ->
      (* A stage boundary in the value language maps onto the ground language's
         own [Round], which already carries f32 semantics. *)
      Ground_expr.Round (recur x)
  | Expr.Value.Select (c, a, b) -> (
      match c with
      (* An index comparison is decided by the coordinate, so the [Select]
         collapses outright rather than surviving as a guard. *)
      | Expr.Bool.Index_eq (x, y) ->
          if Int.equal (index x) (index y) then recur a else recur b
      | Expr.Bool.Value_lt (x, y) ->
          Ground_expr.Select
            (Ground_expr.Lt (recur x, recur y), recur a, recur b))
  | Expr.Value.Load (src, idx) ->
      leaf ~env (Expr_bridge.id_of_source src) (fun a ->
          index (Expr.Coord.get idx a))
  | Expr.Value.Intrinsic i -> max_pool esc ~env ~coord ~rvars i
  | Expr.Value.Reduce r ->
      let lo = index r.Expr.Reduction.lo and hi = index r.Expr.Reduction.hi in
      let combine, seed =
        match r.Expr.Reduction.kind with
        | Expr.Reduction.Sum ->
            ( (fun a b -> Ground_expr.Binary (Expr.Value.Add, a, b)),
              Ground_expr.Const 0. )
        | Expr.Reduction.Max ->
            ( (fun a b -> Ground_expr.Max (Expr.Max_op.Float_max, a, b)),
              Ground_expr.Const neg_infinity )
      in
      (* Same left fold, same seed and same order as [Expr.Eval]'s arm — the
         ground form has to reproduce the engine's association, not merely its
         value. *)
      let rec fold i acc =
        if i >= hi then acc
        else
          fold (i + 1)
            (combine acc
               (ground esc ~env ~coord
                  ~rvars:((r.Expr.Reduction.var, i) :: rvars)
                  r.Expr.Reduction.body))
      in
      fold lo seed

(* A [Load] of a synthetic constant fill is that constant; a [Load] of a bound
   model constant is its stored element, read exactly the way [Expr.Eval.value] reads
   it; anything else is a free cell. *)
and leaf ~env id at_axis : Ground_expr.t =
  let coord =
    Vec6.coord ~n:(at_axis Axis.N) ~t:(at_axis Axis.T) ~d:(at_axis Axis.D)
      ~h:(at_axis Axis.H) ~w:(at_axis Axis.W) ~c:(at_axis Axis.C)
  in
  match
    Option.bind env.Env.constant_store (fun store ->
        Const_ssa_symbolic.ground store id coord)
  with
  | Some expr -> expr
  | None -> (
      match (Env.const_of env id, Env.constant_of env id) with
      | Some v, _ -> Ground_expr.Const v
      | None, Some payload ->
          Ground_expr.Const (Tensor.read_at_raw payload at_axis)
      | None, None ->
          Ground_expr.Cell
            { Ground_expr.Cell.origin = Env.origin env id; coord })

(* The window is concrete, so the stencil expands into the same paired fold
   [Direct]/[Expr.Eval.value] run: one predicate advancing value and index together.
   The value accumulator is a binary [Max] node, so it is mentioned once and
   the fold stays linear in the window; the index accumulator has to name the
   running best inside its guard, which is why it is the larger of the two. *)
and max_pool esc ~env ~coord ~rvars (Expr.Intrinsic.Max_pool d as i) =
  let open Expr.Intrinsic.Max_pool in
  let index : type r. r Expr.Index.t -> int =
   fun x ->
    Expr.Eval.eval_index esc
      ~widen:(fun (e : Expr.Eval.index_error) -> (e :> error))
      ~output:coord
      ~reducers:(fun v ->
        List.find_map
          (fun (w, n) -> if Expr.Reduce_var.equal v w then Some n else None)
          rvars)
      ~resolve_data:(fun src coord ->
        resolve_data_source env (Expr_bridge.id_of_source src) coord)
      x
  in
  (* Through the shared geometry helpers, not recomputed here. [out_h * stride]
     and [ih * in_w] are aggregates of individually valid factors, so they need
     their own bounds -- and a second copy of the arithmetic is exactly how this
     interpreter and [Expr.Eval] would drift apart. *)
  let w =
    or_throw esc
      (Expr.Intrinsic.window i
         ~out_h:(index (Expr.Coord.get d.out Axis.H))
         ~out_w:(index (Expr.Coord.get d.out Axis.W)))
  in
  (* Hoisted out of the fold. [out]'s four non-H/W components do not depend on
     the window position, so re-evaluating them once per window ELEMENT is pure
     waste -- and this sits on the grounding path whose cost the budget comment
     below is about. *)
  let others =
    Expr.Coord.mapi
      (fun a x -> if a = Axis.H || a = Axis.W then 0 else index x)
      d.out
  in
  let read ih iw =
    leaf ~env (Expr_bridge.id_of_source d.source) (fun a ->
        if a = Axis.H then ih
        else if a = Axis.W then iw
        else Expr.Coord.get others a)
  in
  let rec fold_w ih iw (best, best_index) =
    if iw >= w.Expr.Intrinsic.Window.whi then fold_h (ih + 1) (best, best_index)
    else
      let v = read ih iw in
      let flat =
        or_throw esc
          (Expr.Eval.float_of_index
             (or_throw esc (Expr.Intrinsic.flat_index i ~ih ~iw)))
      in
      fold_w ih (iw + 1)
        ( Ground_expr.Max (Expr.Max_op.Pool_max, best, v),
          Ground_expr.Select
            ( Ground_expr.Pool_better { best; value = v },
              Ground_expr.Const flat,
              best_index ) )
  and fold_h ih acc =
    if ih >= w.Expr.Intrinsic.Window.hhi then acc
    else fold_w ih w.Expr.Intrinsic.Window.wlo acc
  in
  let best, best_index =
    fold_h w.Expr.Intrinsic.Window.hlo
      (Ground_expr.Const neg_infinity, Ground_expr.Const 0.)
  in
  match d.result with
  | Expr.Intrinsic.Max_pool.Value -> best
  | Expr.Intrinsic.Max_pool.Index -> best_index

(* ---- the interface -------------------------------------------------------- *)

(* Internal: escapes through [esc]. The public entries below establish it. *)
let body_at esc env (st : Stage_program.Stage.t) coord =
  let limits = Kernel.Limits.default in
  let body =
    or_throw esc
      (Err.map_error
         (fun e -> `Region e)
         (Env.pixel_body env ~max_size:limits.Kernel.Limits.max_size
            ~max_depth:limits.Kernel.Limits.max_depth st))
  in
  ground esc ~env
    ~coord:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int coord))
    ~rvars:[] body

let at env id coord =
  Err.Escape.with_escape @@ fun esc ->
  (* [id] names an edge of THIS graph, so the stage is looked up by that raw id
     DIRECTLY. A root is never replaced by a correspondence variable, whatever
     cluster it is in: doing so would let the cluster under test discharge
     itself by naming both its sides the same thing before either definition was
     looked at. Expanding the root is what puts a real definition on each side;
     projection then applies to what that reads. *)
  match Env.stage_of_id env id with
  | Some st -> Ground_expr.Round (body_at esc env st coord)
  | None -> (
      (* An input edge can itself be a bound constant — [fold_const]'s whole
         output is one — so the same binding [leaf] applies inside a body has to
         apply here too, not just to [Load]s. *)
      match
        Option.bind env.Env.constant_store (fun store ->
            Const_ssa_symbolic.ground store id coord)
      with
      | Some expr -> expr
      | None -> (
          match (Env.const_of env id, Env.constant_of env id) with
          | Some v, _ -> Ground_expr.Const v
          | None, Some payload ->
              Ground_expr.Const
                (Tensor.read_at_raw payload (fun a ->
                     Dim.to_int (Vec6.get coord a)))
          | None, None ->
              let origin = Env.origin env id in
              if Option.is_none (Env.shape_of env origin) then
                or_throw esc (Err.fail (`Unknown_edge id))
              else Ground_expr.Cell { Ground_expr.Cell.origin; coord }))

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
let expand ~boundary ~budget env (e : Ground_expr.t) =
  Err.Escape.with_escape @@ fun esc ->
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
                Ground_expr.Round (body_at esc env st c.Ground_expr.Cell.coord)
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

(* Stays TOTAL, deliberately: it only inspects existing cells and stage
   availability, evaluating no index and no intrinsic, so it has nothing to
   fail on. *)
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
