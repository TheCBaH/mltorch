(* What a statement does to memory and to the invocation, and which buffers may
   share memory.

   Distinct buffer ids are not distinct memory: workspace reuse and borrowed
   views can overlap, and an overlap this library cannot see through is not
   evidence of independence. {!policy} makes the assumption a value a caller
   states: [Conservative] treats every pair of buffers as possibly overlapping,
   [Distinct_buffers] is the guarantee of a caller that has established it (an
   entry check, or a runner that allocates each buffer itself). A pass takes the
   policy it was given; none picks one. *)

type policy = Conservative | Distinct_buffers

let may_overlap policy a b =
  match policy with
  | Conservative -> true
  | Distinct_buffers -> Ssa_id.Buffer.equal a b

type summary = {
  reads : Ssa_id.Buffer.Set.t;
  writes : Ssa_id.Buffer.Set.t;
  may_fail : bool;
      (** a checked operation, check or load that can end the invocation *)
  meter : bool;  (** touches the scan meter *)
  marks : bool;  (** counts logical work *)
  locals : bool;  (** reads or writes a scratch object *)
}

let empty =
  {
    reads = Ssa_id.Buffer.Set.empty;
    writes = Ssa_id.Buffer.Set.empty;
    may_fail = false;
    meter = false;
    marks = false;
    locals = false;
  }

let union a b =
  {
    reads = Ssa_id.Buffer.Set.union a.reads b.reads;
    writes = Ssa_id.Buffer.Set.union a.writes b.writes;
    may_fail = a.may_fail || b.may_fail;
    meter = a.meter || b.meter;
    marks = a.marks || b.marks;
    locals = a.locals || b.locals;
  }

let of_op (op : Ssa_op.t) =
  let read b ~fails =
    { empty with reads = Ssa_id.Buffer.Set.singleton b; may_fail = fails }
  in
  match op with
  | Ssa_op.Load { buffer; _ } -> read buffer ~fails:true
  | Ssa_op.Load_in_bounds { buffer; _ } -> read buffer ~fails:false
  | Ssa_op.Store { buffer; _ } ->
      { empty with writes = Ssa_id.Buffer.Set.singleton buffer }
  | Ssa_op.Check_access { buffer; _ } -> read buffer ~fails:true
  | Ssa_op.Check_gather _ | Ssa_op.Check_local _ | Ssa_op.Check_scan _
  | Ssa_op.Float_to_i64 _ | Ssa_op.I64_div _ | Ssa_op.Index_add _
  | Ssa_op.Index_of_i64 _ | Ssa_op.Index_scale _ ->
      { empty with may_fail = true }
  | Ssa_op.Local_alloc _ | Ssa_op.Local_write _ -> { empty with locals = true }
  | Ssa_op.Local_read _ -> { empty with locals = true; may_fail = true }
  | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ -> { empty with marks = true }
  | Ssa_op.Vec_load { buffer; _ } -> read buffer ~fails:false
  | Ssa_op.Vec_store { buffer; _ } ->
      { empty with writes = Ssa_id.Buffer.Set.singleton buffer }
  | Ssa_op.Meter_charge | Ssa_op.Meter_reserve _ ->
      { empty with meter = true; may_fail = true }
  | Ssa_op.Meter_release _ | Ssa_op.Meter_reset -> { empty with meter = true }
  | Ssa_op.Const _ | Ssa_op.Convert _ | Ssa_op.Float_binary _
  | Ssa_op.Float_compare _ | Ssa_op.Float_fma _ | Ssa_op.Float_max _
  | Ssa_op.Float_unary _ | Ssa_op.I64_arith _ | Ssa_op.I64_compare _
  | Ssa_op.Index_add_in_domain _ | Ssa_op.Index_ceil_div _
  | Ssa_op.Index_clamp_low _ | Ssa_op.Index_compare _ | Ssa_op.Index_floor_div _
  | Ssa_op.Index_max _ | Ssa_op.Index_min _ | Ssa_op.Index_scale_in_domain _
  | Ssa_op.Lanewise _ | Ssa_op.Pool_better _ | Ssa_op.Pred_not _
  | Ssa_op.Pred_or _ | Ssa_op.Select _ | Ssa_op.Vec_extract _
  | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_splat _ ->
      empty

let rec of_stmt : Ssa_region.t Ssa_stmt.t -> summary = function
  | Ssa_stmt.Instr i -> of_op i.Ssa_instr.op
  | Ssa_stmt.For { body; _ } -> of_region body
  | Ssa_stmt.If { then_; else_; _ } -> union (of_region then_) (of_region else_)
  | Ssa_stmt.Ordered_sum { body; _ } -> of_region body

and of_region (r : Ssa_region.t) =
  List.fold_left (fun acc s -> union acc (of_stmt s)) empty r.Ssa_region.body

(* Whether anything a statement does may change what a read of [buffer] sees. *)
let may_write policy (s : summary) buffer =
  Ssa_id.Buffer.Set.exists (fun w -> may_overlap policy w buffer) s.writes
