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
end

(* Two exact accounts over grounding's own construction, both reset per proof
   attempt and both shared by every root registered against one [Meter.t] --
   see the design record's "Grounding meter and verdict mapping". *)
module Budget = struct
  type t = { max_ground_nodes : int64; max_nodes : int }
end

let default_budget =
  { Budget.max_ground_nodes = 2_000_000L; max_nodes = 200_000 }

module Meter = struct
  (* [ground_nodes] is CUMULATIVE construction fuel: every node this module
     builds charges it, including a discarded scan row or a re-embedded
     cached subtree, and it never refunds. [pair_nodes] is the CURRENT total
     logical size of every root presently registered against this meter --
     it goes up and down as [expand] replaces a root's cells, and a root's
     own share is exactly its [Term.size]. *)
  type t = {
    budget : Budget.t;
    mutable ground_nodes : int64;
    mutable pair_nodes : int64;
  }

  let create budget = { budget; ground_nodes = 0L; pair_nodes = 0L }
end

module Term = struct
  (* [size] is cached at construction, never re-derived from [expr] by
     walking it -- that walk is exactly the unmetered rediscovery the design
     record forbids. *)
  type t = { expr : Ground_expr.t; size : int64 }

  let expression t = t.expr
  let size t = t.size
end

(* [Expr.Eval.error] joins the row: grounding evaluates indices, and checked
   arithmetic can now fail where it previously wrapped.
   [`Data_index_unresolved] is [resolve_data_source]'s own conservative
   catch-all (below): a [Data] source this grounder cannot resolve to an
   exact value. It feeds [map_verify_check.ml]'s existing generic
   [Ground_eval.error -> Unproved] conversion unchanged -- an unresolved
   [Data] source makes a cluster [Unproved], never a build failure.
   [`Ground_nodes_over_limit]/[`Pair_nodes_over_limit] carry the configured
   LIMIT, not the observed size, matching this repository's "payload is the
   limit" convention. *)
type error =
  [ Expr.Eval.error
  | `Data_index_unresolved
  | `Ground_nodes_over_limit of int64
  | `Pair_nodes_over_limit of int
  | `Partition of Region_partition.error
  | `Region of Region_program.error
  | `Unknown_edge of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Expr.Eval.error as e -> Expr.Eval.pp_error fmt e
  | `Data_index_unresolved ->
      Fmt.string fmt
        "Data index source could not be resolved to a directly-bound I64 \
         constant"
  | `Ground_nodes_over_limit limit ->
      Fmt.pf fmt "grounding exceeds max_ground_nodes (%Ld)" limit
  | `Pair_nodes_over_limit limit ->
      Fmt.pf fmt "grounding exceeds max_nodes (%d)" limit
  | `Partition e -> Region_partition.pp_error fmt e
  | `Region e -> Region_program.pp_error fmt e
  | `Unknown_edge id -> Fmt.pf fmt "unknown edge %a" Tensor_id.pp id

(* Saturating: matches this repository's 32-bit-safe-aggregate rule
   (js_of_ocaml reaches this library) for every checked size addition below.
   Every operand here is a non-negative node count or limit, so saturation
   at [Int64.max_int] is as sound as failing outright and needs no error
   channel -- the caller compares against a configured limit far below it. *)
let sat_add_i64 a b =
  if Int64.compare a (Int64.sub Int64.max_int b) > 0 then Int64.max_int
  else Int64.add a b

(* Charges [n] construction-fuel nodes against [meter], failing BEFORE the
   caller's node is retained -- this is what stops the blowup the design
   record targets: [ground]/[leaf]/[max_pool] call this at every node they
   build, so a recursive stage-inlining chain cannot construct past the cap
   before anyone gets to measure it. *)
let charge_ground esc (meter : Meter.t) n =
  let total = sat_add_i64 meter.Meter.ground_nodes n in
  if Int64.compare total meter.Meter.budget.Budget.max_ground_nodes > 0 then
    Err.Escape.throw esc
      (`Ground_nodes_over_limit meter.Meter.budget.Budget.max_ground_nodes)
  else meter.Meter.ground_nodes <- total

(* One freshly built node: every [ground]/[leaf]/[max_pool] constructor
   result is wrapped in this, so [Meter.ground_nodes] counts exactly the
   nodes this module allocates -- once each, regardless of how many times a
   caller later reads the resulting value. *)
let node esc meter v =
  charge_ground esc meter 1L;
  v

(* Registers (or re-registers, after [expand] replaces a root's cells) the
   CURRENT total pair size across every root sharing [meter]. [expr]'s
   logical size is measured once, here, on an already construction-fuel-
   bounded tree -- not the unmetered rediscovery the design record forbids,
   since [charge_ground] already proved [expr] is no larger than
   [max_ground_nodes] before this ever walks it. *)
let register esc (meter : Meter.t) delta (expr : Ground_expr.t) : Term.t =
  let total = sat_add_i64 meter.Meter.pair_nodes delta in
  if Int64.compare total (Int64.of_int meter.Meter.budget.Budget.max_nodes) > 0
  then
    Err.Escape.throw esc
      (`Pair_nodes_over_limit meter.Meter.budget.Budget.max_nodes)
  else begin
    meter.Meter.pair_nodes <- total;
    { Term.expr; size = Int64.of_int (Ground_expr.size expr) }
  end

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

(* The local environment a Region program's body is grounded against --
   [Ground_expr.t] counterparts of [Region_execution.evaluate_locals]'s slot
   array, built in declaration order so a later local's body may read an
   earlier one (checked at construction; [Region_program.check] rejects a
   forward reference before grounding ever sees the program). A scalar local
   is one node; a vector local is one node per position; a scan (trace) local
   is one node per (row, lane), flattened row-major exactly as
   [Region_slots]/[Expr.Scan]'s own doc comment specifies -- [width] is
   retained alongside the table only to turn a (row, lane) pair back into a
   flat offset. *)
module Frame = struct
  type t = {
    scalars : Ground_expr.t Expr.Local_var.Map.t;
    vectors : Ground_expr.t array Expr.Local_var.Map.t;
    scans : (Ground_expr.t array * int) Expr.Local_var.Map.t;
  }

  let empty =
    {
      scalars = Expr.Local_var.Map.empty;
      vectors = Expr.Local_var.Map.empty;
      scans = Expr.Local_var.Map.empty;
    }

  let with_scalar t id g =
    { t with scalars = Expr.Local_var.Map.add id g t.scalars }

  let with_vector t id arr =
    { t with vectors = Expr.Local_var.Map.add id arr t.vectors }

  let with_scan t id table width =
    { t with scans = Expr.Local_var.Map.add id (table, width) t.scans }

  (* [prev]'s scope is exactly one scan's own [update] evaluation -- overriding
     just its entry in [vectors], never touching [t]'s own bindings, is what
     lets every OTHER local reference in [update] keep resolving through the
     enclosing frame unchanged, mirroring [Expr.Eval.value]'s [local_at_ref]
     override for the same binder. *)
  let with_prev t prev row = with_vector t prev row
end

(* [Local_scan_at]/[Scan_at] bounds failures reuse [Expr.Eval]'s own
   [scan_error] vocabulary -- the same tags [Region_eval]/[Region_execution]
   report for exactly the same conditions, so a caller sees one error
   vocabulary for a trace-shape violation regardless of which evaluator
   caught it. *)
let scan_bounds ~local ~row ~lane ~extent kind =
  let projection = { Expr.Eval.Scan_projection.local; row; lane } in
  `Scan_projection (kind { Expr.Eval.Scan_bounds.projection; extent })

(* [Load]s become leaves through [leaf]; every index is evaluated at [coord]
   (and at the enclosing reduction variables), which is what removes the
   binders. Every node this builds is interned into [arena] -- the current
   root lineage's own raw arena, so a recurrence's or a repeated max-pool
   accumulator's structurally-equal subterms share one node instead of
   duplicating it. [frame] resolves [Local]/[Local_at]/[Local_scan_at] against
   the enclosing Region program's already-grounded locals; a legacy Pixel
   stage (no locals at all) grounds with [Frame.empty], so this is the SAME
   traversal for both, never a special-cased Pixel path. *)
let rec ground esc ~env ~meter ~arena ~frame ~coord ~rvars (e : Expr.Value.t) :
    Ground_expr.t =
  let recur = ground esc ~env ~meter ~arena ~frame ~coord ~rvars in
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
  | Expr.Value.Const x -> node esc meter (Ground_expr.const arena x)
  | Expr.Value.Local v -> (
      match Expr.Local_var.Map.find_opt v frame.Frame.scalars with
      | Some g -> g
      | None -> Err.Escape.throw esc (`Unbound_local v))
  | Expr.Value.Local_at (v, i) -> (
      let pos = index i in
      match Expr.Local_var.Map.find_opt v frame.Frame.vectors with
      | Some arr when pos >= 0 && pos < Array.length arr -> arr.(pos)
      | Some _ | None -> Err.Escape.throw esc (`Unbound_local v))
  | Expr.Value.Local_scan_at (v, row_i, lane_i) -> (
      let row = index row_i and lane = index lane_i in
      match Expr.Local_var.Map.find_opt v frame.Frame.scans with
      | None ->
          Err.Escape.throw esc (`Scan_projection (Expr.Eval.Unknown_local v))
      | Some (table, width) ->
          if row < 0 || row * width >= Array.length table then
            Err.Escape.throw esc
              (scan_bounds ~local:(Some v) ~row ~lane
                 ~extent:(Array.length table / width)
                 (fun b -> Expr.Eval.Row_out_of_range b))
          else if lane < 0 || lane >= width then
            Err.Escape.throw esc
              (scan_bounds ~local:(Some v) ~row ~lane ~extent:width (fun b ->
                   Expr.Eval.Lane_out_of_range b))
          else table.((row * width) + lane))
  | Expr.Value.Scan_at (s, row_i, lane_i) ->
      let row = index row_i and lane = index lane_i in
      ground_scan_at esc ~env ~meter ~arena ~frame ~coord ~rvars s ~row ~lane
  | Expr.Value.Binary (op, a, b) ->
      node esc meter (Ground_expr.binary arena op (recur a) (recur b))
  | Expr.Value.Unary (op, x) ->
      node esc meter (Ground_expr.unary arena op (recur x))
  | Expr.Value.Value_of_index i ->
      (* Through the shared conversion, not [float_of_int]: this is the second
         interpreter, and a helper returning [int] does not make the conversion
         that follows it exact. *)
      node esc meter
        (Ground_expr.const arena
           (or_throw esc (Expr.Eval.float_of_index (index i))))
  | Expr.Value.Round_f32 x ->
      (* A stage boundary in the value language maps onto the ground language's
         own [Round], which already carries f32 semantics. *)
      node esc meter (Ground_expr.round arena (recur x))
  | Expr.Value.Select (c, a, b) -> (
      match c with
      (* An index comparison is decided by the coordinate, so the [Select]
         collapses outright rather than surviving as a guard. *)
      | Expr.Bool.Index_eq (x, y) ->
          if Int.equal (index x) (index y) then recur a else recur b
      | Expr.Bool.Value_lt (x, y) ->
          node esc meter
            (Ground_expr.select arena
               (Ground_expr.lt arena (recur x) (recur y))
               (recur a) (recur b)))
  | Expr.Value.Load (src, idx) ->
      leaf esc ~env ~meter ~arena (Expr_bridge.id_of_source src) (fun a ->
          index (Expr.Coord.get idx a))
  | Expr.Value.Intrinsic i -> max_pool esc ~env ~meter ~arena ~coord ~rvars i
  | Expr.Value.Reduce r ->
      let lo = index r.Expr.Reduction.lo and hi = index r.Expr.Reduction.hi in
      let combine, seed =
        match r.Expr.Reduction.kind with
        | Expr.Reduction.Sum ->
            ( (fun a b ->
                node esc meter (Ground_expr.binary arena Expr.Value.Add a b)),
              node esc meter (Ground_expr.const arena 0.) )
        | Expr.Reduction.Max ->
            ( (fun a b ->
                node esc meter (Ground_expr.max arena Expr.Max_op.Float_max a b)),
              node esc meter (Ground_expr.const arena neg_infinity) )
      in
      (* Same left fold, same seed and same order as [Expr.Eval]'s arm — the
         ground form has to reproduce the engine's association, not merely its
         value. *)
      let rec fold i acc =
        if i >= hi then acc
        else
          fold (i + 1)
            (combine acc
               (ground esc ~env ~meter ~arena ~frame ~coord
                  ~rvars:((r.Expr.Reduction.var, i) :: rvars)
                  r.Expr.Reduction.body))
      in
      fold lo seed

(* Inline [Scan_at]: [row]/[lane] are already evaluated (the caller needs them
   to key a bounds error against the RIGHT descriptor, [None] rather than
   [Some id] -- see [Expr.Eval.Scan_projection.local]). Builds the whole
   prefix from row 0 up to [row] fresh, exactly mirroring
   [Expr_internal.Eval.eval_scan_at]'s two-buffer loop but producing
   [Ground_expr.t] nodes: [init] grounds each lane once with [lane] bound;
   each later row grounds [update] with [lane]/[step] bound and [prev]
   resolved by overriding just that one binder in [frame] (never a mutable
   ref -- grounding is a pure recursive builder, so there is no unwind to
   protect). Hash-consing already deduplicates a node this rebuilds against
   an identical earlier construction call within the same arena; what is NOT
   cached across separate [Scan_at] occurrences is the CONSTRUCTION WORK
   itself (each occurrence re-walks and re-charges fuel for its own prefix) --
   a disclosed performance gap relative to the design record's per-instance
   prefix cache, not a correctness one. *)
and ground_scan_at esc ~env ~meter ~arena ~frame ~coord ~rvars (s : Expr.Scan.t)
    ~row ~lane : Ground_expr.t =
  if row < 0 || row > s.Expr.Scan.steps then
    Err.Escape.throw esc
      (scan_bounds ~local:None ~row ~lane ~extent:(s.Expr.Scan.steps + 1)
         (fun b -> Expr.Eval.Row_out_of_range b))
  else if lane < 0 || lane >= s.Expr.Scan.width then
    Err.Escape.throw esc
      (scan_bounds ~local:None ~row ~lane ~extent:s.Expr.Scan.width (fun b ->
           Expr.Eval.Lane_out_of_range b))
  else
    let width = s.Expr.Scan.width in
    let init_row () =
      Array.init width (fun l ->
          ground esc ~env ~meter ~arena ~frame ~coord
            ~rvars:((s.Expr.Scan.lane, l) :: rvars)
            s.Expr.Scan.init)
    in
    let next_row ~step prev_row =
      let frame' = Frame.with_prev frame s.Expr.Scan.prev prev_row in
      Array.init width (fun l ->
          ground esc ~env ~meter ~arena ~frame:frame' ~coord
            ~rvars:((s.Expr.Scan.lane, l) :: (s.Expr.Scan.step, step) :: rvars)
            s.Expr.Scan.update)
    in
    let rec run r prev_row =
      if r = row then prev_row.(lane)
      else run (r + 1) (next_row ~step:r prev_row)
    in
    run 0 (init_row ())

(* The full trace of a Region-authored scan LOCAL, eager and row-major --
   [Region_execution.evaluate_locals]'s own contract for a trace local,
   reproduced here as [Ground_expr.t] nodes rather than floats. Every row
   after the first overrides [prev] in [frame] with the PREVIOUS row alone (an
   [Array.sub] slice -- a small, deliberate allocation per row, not a
   correctness concern), the same override [ground_scan_at]'s own [next_row]
   uses. Not shared code with it: this fills every row unconditionally (no
   target row to stop at) and returns the whole table for the enclosing
   [Frame.t], where [ground_scan_at] returns one cell. *)
and ground_scan_local esc ~env ~meter ~arena ~frame ~coord (s : Expr.Scan.t) :
    Ground_expr.t array =
  let width = s.Expr.Scan.width and steps = s.Expr.Scan.steps in
  let table = Array.make ((steps + 1) * width) (Ground_expr.const arena 0.) in
  for l = 0 to width - 1 do
    table.(l) <-
      ground esc ~env ~meter ~arena ~frame ~coord
        ~rvars:[ (s.Expr.Scan.lane, l) ]
        s.Expr.Scan.init
  done;
  for r = 1 to steps do
    let prev_row = Array.sub table ((r - 1) * width) width in
    let frame' = Frame.with_prev frame s.Expr.Scan.prev prev_row in
    for l = 0 to width - 1 do
      table.((r * width) + l) <-
        ground esc ~env ~meter ~arena ~frame:frame' ~coord
          ~rvars:[ (s.Expr.Scan.lane, l); (s.Expr.Scan.step, r - 1) ]
          s.Expr.Scan.update
    done
  done;
  table

(* A [Load] of a synthetic constant fill is that constant; a [Load] of a bound
   model constant is its stored element, read exactly the way [Expr.Eval.value] reads
   it; anything else is a free cell. A Const-SSA-backed capture is the one case
   that hands back an ALREADY BUILT subtree rather than one node built here, so
   it charges that subtree's whole measured size instead of [node]'s flat 1. *)
and leaf esc ~env ~meter ~arena id at_axis : Ground_expr.t =
  let coord =
    Vec6.coord ~n:(at_axis Axis.N) ~t:(at_axis Axis.T) ~d:(at_axis Axis.D)
      ~h:(at_axis Axis.H) ~w:(at_axis Axis.W) ~c:(at_axis Axis.C)
  in
  match
    Option.bind env.Env.constant_store (fun store ->
        Const_ssa_symbolic.ground arena store id coord)
  with
  | Some expr ->
      charge_ground esc meter (Int64.of_int (Ground_expr.size expr));
      expr
  | None -> (
      match (Env.const_of env id, Env.constant_of env id) with
      | Some v, _ -> node esc meter (Ground_expr.const arena v)
      | None, Some payload ->
          node esc meter
            (Ground_expr.const arena (Tensor.read_at_raw payload at_axis))
      | None, None ->
          node esc meter
            (Ground_expr.cell arena
               { Ground_expr.Cell.origin = Env.origin env id; coord }))

(* The window is concrete, so the stencil expands into the same paired fold
   [Direct]/[Expr.Eval.value] run: one predicate advancing value and index together.
   The value accumulator is a binary [Max] node, so it is mentioned once and
   the fold stays linear in the window; the index accumulator has to name the
   running best inside its guard, which is why it is the larger of the two. *)
and max_pool esc ~env ~meter ~arena ~coord ~rvars
    (Expr.Intrinsic.Max_pool d as i) =
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
    leaf esc ~env ~meter ~arena (Expr_bridge.id_of_source d.source) (fun a ->
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
        ( node esc meter (Ground_expr.max arena Expr.Max_op.Pool_max best v),
          node esc meter
            (Ground_expr.select arena
               (Ground_expr.pool_better arena ~best ~value:v)
               (node esc meter (Ground_expr.const arena flat))
               best_index) )
  and fold_h ih acc =
    if ih >= w.Expr.Intrinsic.Window.hhi then acc
    else fold_w ih w.Expr.Intrinsic.Window.wlo acc
  in
  let best, best_index =
    fold_h w.Expr.Intrinsic.Window.hlo
      ( node esc meter (Ground_expr.const arena neg_infinity),
        node esc meter (Ground_expr.const arena 0.) )
  in
  match d.result with
  | Expr.Intrinsic.Max_pool.Value -> best
  | Expr.Intrinsic.Max_pool.Index -> best_index

(* ---- the interface -------------------------------------------------------- *)

(* Internal: escapes through [esc]. The public entries below establish it.
   Grounds the stage's own [Region_program.t] DIRECTLY -- never through
   [Region_program.specialize_pixel], which would inline every trace read as a
   re-executing [Scan_at] and re-embed the whole prior-step subtree at every
   later step -- the same "no expression-level sharing duplicates the whole
   subtree" hazard CLAUDE.md's construction rules flag elsewhere, here at the
   grounding layer. [check]/[preflight] run with the same shape and limits
   [Region_execution.validate] uses, so grounding rejects the same programs
   Region execution would.

   Locals are grounded ONCE, at the Region KEY [coord] maps to -- exactly the
   coordinate [Region_execution.evaluate_locals] would fill one slot array
   for -- in declaration order, each added to [frame] before the next local's
   body can reference it. The stage's OWN output expression is then grounded
   at the full requested [coord] (which may differ from the key on a
   non-singleton axis), reading the frame [Region_execution.emit] would read
   from the same slot array. A legacy Pixel stage's program has no locals and
   a singleton partition, so it takes this exact path with an empty frame and
   [key = coord] -- there is no separate Pixel case here at all. *)
let body_at esc env ~meter ~arena (st : Stage_program.Stage.t) coord =
  let limits = Kernel.Limits.default in
  let max_size = limits.Kernel.Limits.max_size
  and max_depth = limits.Kernel.Limits.max_depth in
  let scan_limits = Kernel.Limits.scan_limits limits in
  let program = Stage_program.Stage.computation st in
  let region e = Err.map_error (fun e -> `Region e) e in
  or_throw esc (region (Region_program.check ~max_size ~max_depth program));
  or_throw esc
    (region
       (Region_program.preflight
          ~max_local_slots:limits.Kernel.Limits.max_local_slots
          ~max_scan_state:(Expr.Scan_limits.max_state scan_limits)
          ~max_scan_updates:(Expr.Scan_limits.max_updates scan_limits)
          ~output_shape:st.Stage_program.Stage.sg.Tensor_sig.shape program));
  let key =
    or_throw esc
      (Err.map_error
         (fun e -> `Partition e)
         (Region_partition.key_of_output
            ~output_shape:st.Stage_program.Stage.sg.Tensor_sig.shape
            (Region_program.partition program)
            coord))
  in
  let key_coord = Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int key) in
  let frame =
    List.fold_left
      (fun frame (local : Region_local.t) ->
        match local.Region_local.rhs with
        | Region_local.Rhs.Scalar value ->
            let g =
              ground esc ~env ~meter ~arena ~frame ~coord:key_coord ~rvars:[]
                value
            in
            Frame.with_scalar frame local.Region_local.id g
        | Region_local.Rhs.Vector { extent; var; body } ->
            let arr =
              Array.init extent (fun p ->
                  ground esc ~env ~meter ~arena ~frame ~coord:key_coord
                    ~rvars:[ (var, p) ]
                    body)
            in
            Frame.with_vector frame local.Region_local.id arr
        | Region_local.Rhs.Scan s ->
            let table =
              ground_scan_local esc ~env ~meter ~arena ~frame ~coord:key_coord s
            in
            Frame.with_scan frame local.Region_local.id table s.Expr.Scan.width)
      Frame.empty
      (Region_program.locals program)
  in
  ground esc ~env ~meter ~arena ~frame
    ~coord:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int coord))
    ~rvars:[]
    (Region_program.output program)

(* Registers a NEW root, in a freshly allocated arena that belongs to it for
   the rest of its lifetime -- including every later [expand] on the [Term.t]
   this returns. A separate arena per registered root (rather than one shared
   across every root [meter] ever sees) is what keeps [expand]'s "everything
   else registered against [meter]" arithmetic sound: two roots can never
   reach a shared node, so one root's [Ground_expr.size] is never someone
   else's contribution counted twice. Its whole measured size is added to
   [meter]'s current pair total, on top of whatever [Meter.ground_nodes]
   construction already charged while building it. *)
let at ~meter env id coord =
  Err.Escape.with_escape @@ fun esc ->
  let arena = Ground_expr.Arena.create () in
  (* [id] names an edge of THIS graph, so the stage is looked up by that raw id
     DIRECTLY. A root is never replaced by a correspondence variable, whatever
     cluster it is in: doing so would let the cluster under test discharge
     itself by naming both its sides the same thing before either definition was
     looked at. Expanding the root is what puts a real definition on each side;
     projection then applies to what that reads. *)
  let register_new expr =
    register esc meter (Int64.of_int (Ground_expr.size expr)) expr
  in
  match Env.stage_of_id env id with
  | Some st ->
      register_new
        (node esc meter
           (Ground_expr.round arena (body_at esc env ~meter ~arena st coord)))
  | None -> (
      (* An input edge can itself be a bound constant — [fold_const]'s whole
         output is one — so the same binding [leaf] applies inside a body has to
         apply here too, not just to [Load]s. *)
      match
        Option.bind env.Env.constant_store (fun store ->
            Const_ssa_symbolic.ground arena store id coord)
      with
      | Some expr ->
          charge_ground esc meter (Int64.of_int (Ground_expr.size expr));
          register_new expr
      | None -> (
          match (Env.const_of env id, Env.constant_of env id) with
          | Some v, _ ->
              register_new (node esc meter (Ground_expr.const arena v))
          | None, Some payload ->
              register_new
                (node esc meter
                   (Ground_expr.const arena
                      (Tensor.read_at_raw payload (fun a ->
                           Dim.to_int (Vec6.get coord a)))))
          | None, None ->
              let origin = Env.origin env id in
              if Option.is_none (Env.shape_of env origin) then
                or_throw esc (Err.fail (`Unknown_edge id))
              else
                register_new
                  (node esc meter
                     (Ground_expr.cell arena { Ground_expr.Cell.origin; coord }))
          ))

(* [meter.Meter.budget.max_nodes] bounds the CURRENT total pair size across
   every root sharing [meter], and has to be checked per replacement: a single
   substitution step is quadratic where a conv feeds a conv, so a term can
   reach tens of millions of nodes before anyone gets to measure it -- and
   measuring only afterward once made verifying a real model cost 25x the
   transform it was checking. [meter.Meter.ground_nodes] (charged by [node]/
   [charge_ground] as [body_at] builds each replacement's own subtree, exactly
   as [ground] does at every other construction site) is the complementary,
   CUMULATIVE account: it catches a round that replaces many modestly-sized
   cells, which the pair total alone would not stop until the very last one.

   A replacement that would cross [max_nodes] is skipped -- the cell stays a
   [Cell], keeping [expandable] true so the driver reports a budget verdict --
   but the walk continues to other cells, so a large chain elsewhere in the
   same round does not stop a small, affordable one from closing. Running out
   leaves the remaining cells unexpanded, which is sound rather than
   approximate: no probe may run against a frontier that never reached the
   inputs.

   [boundary] stops it at the LOCAL frontier: a cell whose cluster gives it a
   variable is a free variable of this obligation, so expanding through it
   would be re-proving someone else's. That is the difference between a
   budget truncation and a completed frontier, and the two must not be
   confused — which is why [expandable] below asks the same question. *)
let expand ~meter ~boundary env (term : Term.t) =
  Err.Escape.with_escape @@ fun esc ->
  (* Reuses the root's OWN arena, allocated once by [at] -- a replacement's
     subtree hash-conses against everything already in the term, and the
     [Term.t] this returns keeps referring to the same root lineage. *)
  let arena = Ground_expr.arena (Term.expression term) in
  let cap = Int64.of_int meter.Meter.budget.Budget.max_nodes in
  (* What every OTHER root registered against [meter] currently contributes;
     [term]'s own share is replaced wholesale below rather than accumulated
     onto. *)
  let others = Int64.sub meter.Meter.pair_nodes (Term.size term) in
  let rec go total e =
    if Int64.compare total cap >= 0 then (total, e)
    else
      match Ground_expr.out e with
      | Ground_expr.Cell c
        when Option.is_some (boundary c.Ground_expr.Cell.origin) ->
          (total, e)
      | Ground_expr.Cell c -> (
          match Env.stage_of env c.Ground_expr.Cell.origin with
          | Some st ->
              let body =
                node esc meter
                  (Ground_expr.round arena
                     (body_at esc env ~meter ~arena st c.Ground_expr.Cell.coord))
              in
              (* Replacing a one-node cell with [body] costs [size body - 1],
                 per the design record's account table -- the cell's own unit
                 is already part of [total]. *)
              let candidate =
                sat_add_i64 total
                  (Int64.sub (Int64.of_int (Ground_expr.size body)) 1L)
              in
              if Int64.compare candidate cap > 0 then (total, e)
              else (candidate, body)
          | None -> (total, e))
      | Ground_expr.Const _ -> (total, e)
      | Ground_expr.Binary (op, a, b) ->
          let total, a = go total a in
          let total, b = go total b in
          (total, Ground_expr.binary arena op a b)
      | Ground_expr.Max (op, a, b) ->
          let total, a = go total a in
          let total, b = go total b in
          (total, Ground_expr.max arena op a b)
      | Ground_expr.Round x ->
          let total, x = go total x in
          (total, Ground_expr.round arena x)
      | Ground_expr.Unary (op, x) ->
          let total, x = go total x in
          (total, Ground_expr.unary arena op x)
      | Ground_expr.Select (g, a, b) ->
          let total, g =
            match Ground_expr.guard_out g with
            | Ground_expr.Lt (x, y) ->
                let total, x = go total x in
                let total, y = go total y in
                (total, Ground_expr.lt arena x y)
            | Ground_expr.Pool_better { best; value } ->
                let total, best = go total best in
                let total, value = go total value in
                (total, Ground_expr.pool_better arena ~best ~value)
          in
          let total, a = go total a in
          let total, b = go total b in
          (total, Ground_expr.select arena g a b)
  in
  let total, e' = go meter.Meter.pair_nodes (Term.expression term) in
  meter.Meter.pair_nodes <- total;
  { Term.expr = e'; size = Int64.sub total others }

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
