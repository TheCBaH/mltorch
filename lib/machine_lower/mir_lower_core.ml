(* The lowering's state and the primitive emitters every SSA operation's
   expansion is built from: refusals, mutations, value binding, constants,
   guards that end a block, and the guarded coordinate address. [Mir_lower]
   includes this module; its interface is the only published one. *)

open Ssa_ir
open Machine_ir
module B = Mir_builder
module L = Mir_layout_map

module Refusal = struct
  type t =
    | Buffer_layout of Ssa_id.Buffer.t
    | Cfg of Ssa_cfg_lower.error
    | Invalid_cfg of Ssa_cfg_verify.diagnostic
    | Invalid_lowering of Mir_diagnostic.t
    | Invalid_program of Ssa_verify.diagnostic
    | Local_object of Ssa_id.Value.t
    | Local_storage of Ssa_id.Value.t
    | Operation of { op : string; slice : Mir_census.Slice.t }
    | Planning of Mir_planning.mismatch
    | Precision of Mir_planning.Precision.t
    | Type of { ty : Ssa_type.t; slice : Mir_census.Slice.t }

  let pp fmt = function
    | Buffer_layout b ->
        Fmt.pf fmt "buffer %a has no representable size" Ssa_id.Buffer.pp b
    | Cfg e -> Ssa_cfg_lower.pp_error fmt e
    | Invalid_cfg d -> Ssa_cfg_verify.pp_error fmt (`Invalid_cfg d)
    | Invalid_lowering d -> Fmt.pf fmt "lowering defect: %a" Mir_diagnostic.pp d
    | Invalid_program d -> Ssa_verify.pp_diagnostic fmt d
    | Local_object v ->
        Fmt.pf fmt "local %a is not its allocation's own result" Ssa_id.Value.pp
          v
    | Local_storage v ->
        Fmt.pf fmt "local %a takes local storage past %Ld bytes" Ssa_id.Value.pp
          v L.Scratch.local_limit
    | Operation { op; slice } ->
        Fmt.pf fmt "%s is admitted by %s" op (Mir_census.Slice.name slice)
    | Planning m -> Mir_planning.pp_mismatch fmt m
    | Precision p ->
        Fmt.pf fmt "binary32 arithmetic under a %s summary"
          (Mir_planning.Precision.name p)
    | Type { ty; slice } ->
        Fmt.pf fmt "type %a is admitted by %s" Ssa_type.pp ty
          (Mir_census.Slice.name slice)
end

module Mutation = struct
  type t =
    | Channel_zero
    | Charge_after_body
    | Conversion_order
    | Double_rounding
    | Eager_load
    | Erf_distributed
    | Erf_single_rounding
    | Guard_order
    | Operand_order
    | Scale_bytes
    | Sequential_transfer
    | Stale_local
    | Vector_stride  (** a lane access's stride doubled *)
    | Zero_extend
end

(* A local allocation site's object: its cell count and variable. *)
module Local_object = struct
  type t = { slots : int64; var : Expr.Local_var.t option }
end

(* The lowering state of one function. *)
type st = {
  esc : Refusal.t Err.Escape.t;
  bld : B.t;
  layout : L.Entry.t list;
  values : (int, Mir_value.t) Hashtbl.t;  (** SSA value id -> machine value *)
  constants : (int, Mir_const.t) Hashtbl.t;
      (** machine value id -> the constant it was emitted as *)
  heads : (int, B.block) Hashtbl.t;
      (** CFG block id -> its first machine block *)
  locals : (int, Local_object.t) Hashtbl.t;
      (** an allocation's SSA result id -> its site's object *)
  mutable objects : (Mir_region.t * Mir_view.t) list;
      (** local sites' storage, latest first *)
  mutable local_bytes : int64;
  limits : Expr.Scan_limits.t;
  mutable deferred_charges : int;  (** under [Charge_after_body] only *)
  mutable helpers : Mir_math.Fn.t list;  (** the math helpers called *)
  mutable cur : B.block;
  mutable origin : Mir_origin.t;
  mutation : Mutation.t option;
}

let mutated st m = st.mutation = Some m
let refuse st r = Err.Escape.throw st.esc r

let machine_type st (ty : Ssa_type.t) =
  match Mir_census.machine_type ty with
  | Ok t -> t
  | Error slice -> refuse st (Refusal.Type { ty; slice })

let is_effect (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

let value st (v : Ssa_value.t) =
  match Hashtbl.find_opt st.values (v.Ssa_value.id :> int) with
  | Some m -> m
  | None -> invalid_arg "Mir_lower: a value used before its definition"

let bind st (v : Ssa_value.t) m =
  Hashtbl.replace st.values (v.Ssa_value.id :> int) m

let with_role st role = st.origin <- { st.origin with Mir_origin.role }
let emit st op = B.emit ~origin:st.origin st.bld st.cur op
let emit_unit st op = B.emit_unit ~origin:st.origin st.bld st.cur op

let const st c =
  let v = emit st (Mir_op.Const c) in
  Hashtbl.replace st.constants (Mir_id.Value.to_int v.Mir_value.id) c;
  v

let is_zero st (v : Mir_value.t) =
  match Hashtbl.find_opt st.constants (Mir_id.Value.to_int v.Mir_value.id) with
  | Some c -> Int64.equal c.Mir_const.bits 0L
  | None -> false

let i32k st x = const st (Mir_const.i32 x)
let i64k st x = const st (Mir_const.i64 x)

let sext st v =
  let k =
    if mutated st Mutation.Zero_extend then Mir_op.Iext.Zext
    else Mir_op.Iext.Sext
  in
  emit st (Mir_op.Iext (k, Mir_width.W64, v))

let icmp st c a b = emit st (Mir_op.Icmp (c, a, b))
let pand st a b = emit st (Mir_op.Pbinary (Mir_op.Pbinary.And, a, b))
let select st p a b = emit st (Mir_op.Select (p, a, b))

(* Ends the current block on [ok]: true continues in a fresh block, false
   reaches a fresh block that computes the payload and fails. *)
let guard st ~ok failure payload =
  let role = st.origin.Mir_origin.role in
  let cont = B.new_block st.bld [] and bad = B.new_block st.bld [] in
  B.branch st.cur ok (cont, []) (bad, []);
  st.cur <- bad;
  with_role st Mir_origin.Role.Payload;
  let p = payload () in
  B.fail ~origin:st.origin st.cur failure p;
  st.cur <- cont;
  with_role st role

let pnot st a = emit st (Mir_op.Pnot a)
let fcmp st c a b = emit st (Mir_op.Fcmp (c, a, b))
let f64k st x = const st (Mir_const.f64 x)
let index_min = -0x8000_0000L
let index_max = 0x7FFF_FFFFL

(* An i64 inside the index domain. *)
let in_index_domain st x =
  let lo = icmp st Mir_op.Icmp.Sle (i64k st index_min) x in
  let hi = icmp st Mir_op.Icmp.Sle x (i64k st index_max) in
  pand st lo hi

let entry_of st id =
  match L.find st.layout id with
  | Some e -> e
  | None -> invalid_arg "Mir_lower: an undeclared buffer"

(* The axis guards of a checked coordinate access, in axis order: the first
   axis outside fails with every coordinate. *)
let coord_guards st (e : L.Entry.t) (c : Ssa_value.t Expr.Coord.t) =
  let extents = e.L.Entry.buffer.Ssa_buffer.extents in
  let source = Ssa_buffer.source e.L.Entry.buffer in
  with_role st Mir_origin.Role.Guard;
  List.iter
    (fun axis ->
      let x = value st (Expr.Coord.get c axis) in
      let ext = Expr.Coord.get extents axis in
      let ok =
        pand st
          (icmp st Mir_op.Icmp.Sle (i32k st 0L) x)
          (icmp st Mir_op.Icmp.Slt x (i32k st ext))
      in
      guard st ~ok
        (Mir_failure.Coord_out_of_range { Mir_failure.Coord.source; axis })
        (fun () ->
          List.map
            (fun a -> sext st (value st (Expr.Coord.get c a)))
            Expr.Axis.all))
    (if mutated st Mutation.Guard_order then List.rev Expr.Axis.all
     else Expr.Axis.all)

(* The address of an access: the view's base plus the row-major element offset
   times the element bytes, all in i64. A flat offset is an element index.
   The row-major fold leaves out what is exactly zero or one: a coordinate
   emitted as the constant 0 adds nothing (and while every earlier one is, the
   offset is still zero), and an extent of 1 multiplies by nothing. *)
let address st (e : L.Entry.t) (at : Ssa_access.t) =
  with_role st Mir_origin.Role.Address;
  let element =
    match at with
    | Ssa_access.Flat v -> sext st (value st v)
    | Ssa_access.Coord c -> (
        let extents = e.L.Entry.buffer.Ssa_buffer.extents in
        List.fold_left
          (fun acc axis ->
            let x = value st (Expr.Coord.get c axis) in
            let zero = is_zero st x in
            match acc with
            | None -> if zero then None else Some (sext st x)
            | Some acc ->
                let ext = Expr.Coord.get extents axis in
                let scaled =
                  if Int64.equal ext 1L then acc
                  else
                    emit st
                      (Mir_op.Iarith (Mir_op.Iarith.Mul, acc, i64k st ext))
                in
                if zero then Some scaled
                else
                  Some
                    (emit st
                       (Mir_op.Iarith (Mir_op.Iarith.Add, scaled, sext st x))))
          None Expr.Axis.all
        |> function
        | Some x -> x
        | None -> i64k st 0L)
  in
  let scale =
    if mutated st Mutation.Scale_bytes then Int64.mul 2L e.L.Entry.elem_bytes
    else e.L.Entry.elem_bytes
  in
  let bytes =
    emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, element, i64k st scale))
  in
  let base = emit st (Mir_op.Addr e.L.Entry.view) in
  emit st (Mir_op.Ptr_add (base, bytes))

let unsupported st op =
  let r = Mir_census.op_row op in
  refuse st
    (Refusal.Operation
       { op = r.Mir_census.Row.op; slice = r.Mir_census.Row.slice })
