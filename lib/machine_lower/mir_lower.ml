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
    | Conversion_order
    | Double_rounding
    | Eager_load
    | Guard_order
    | Operand_order
    | Scale_bytes
    | Sequential_transfer
    | Zero_extend
end

type result = {
  program : Mir_verify.Generic.t;
  layout : L.Entry.t list;
  planning : Mir_planning.t;
}

let subject p = Digest.to_hex (Digest.string (Ssa_pp.to_string p))

let summary ?target ?fma (plan : Ssa_plan.t) =
  let contraction = Ssa_numerics.contraction_permitted plan.Ssa_plan.numerics in
  let fma =
    match fma with
    | Some f -> f
    | None -> (
        match target with
        | Some (t : Ssa_target.t) when contraction && t.Ssa_target.fma ->
            Mir_planning.Fma.Exact
        | Some t when contraction && t.Ssa_target.relaxed_madd ->
            Mir_planning.Fma.Relaxed_madd
        | Some _ | None -> Mir_planning.Fma.Forbidden)
  in
  let precision =
    match plan.Ssa_plan.precision with
    | Ssa_numerics.Precision.F32 -> Mir_planning.Precision.F32
    | Ssa_numerics.Precision.F64 -> Mir_planning.Precision.F64
  in
  let lanes, schedule, capabilities =
    match target with
    | None -> (Mir_type.Lanes.of_int 1, "scalar", [])
    | Some t ->
        ( Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int t.Ssa_target.lanes),
          Printf.sprintf "%s/row_block=%d" t.Ssa_target.name
            t.Ssa_target.row_block,
          Mir_planning.Capability.Vector_bits
            (Int64.of_int t.Ssa_target.vector_bits)
          ::
          (if fma = Mir_planning.Fma.Exact then
             [ Mir_planning.Capability.Fused_multiply_add ]
           else []) )
  in
  Mir_planning.make
    ~subject:(subject plan.Ssa_plan.program)
    ~policy:(Ssa_numerics.identity plan.Ssa_plan.numerics)
    ~schedule ~precision ~lanes ~fma ~capabilities

(* The lowering state of one function. *)
type st = {
  esc : Refusal.t Err.Escape.t;
  bld : B.t;
  layout : L.Entry.t list;
  values : (int, Mir_value.t) Hashtbl.t;  (** SSA value id -> machine value *)
  heads : (int, B.block) Hashtbl.t;
      (** CFG block id -> its first machine block *)
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
let const st c = emit st (Mir_op.Const c)
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
  B.fail st.cur failure p;
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
   times the element bytes, all in i64. A flat offset is an element index. *)
let address st (e : L.Entry.t) (at : Ssa_access.t) =
  with_role st Mir_origin.Role.Address;
  let element =
    match at with
    | Ssa_access.Flat v -> sext st (value st v)
    | Ssa_access.Coord c ->
        let extents = e.L.Entry.buffer.Ssa_buffer.extents in
        List.fold_left
          (fun acc axis ->
            let x = sext st (value st (Expr.Coord.get c axis)) in
            match acc with
            | None -> Some x
            | Some acc ->
                let scaled =
                  emit st
                    (Mir_op.Iarith
                       ( Mir_op.Iarith.Mul,
                         acc,
                         i64k st (Expr.Coord.get extents axis) ))
                in
                Some (emit st (Mir_op.Iarith (Mir_op.Iarith.Add, scaled, x))))
          None Expr.Axis.all
        |> Option.get
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

let decode st op addr (d : Ssa_op.Decode.t) =
  with_role st Mir_origin.Role.Decode;
  let load width align =
    emit st (Mir_op.Load { Mir_op.Access.width; addr; align })
  in
  match d with
  | Ssa_op.Decode.F32_to_f64 ->
      let bits = load Mir_width.W32 4L in
      let f = emit st (Mir_op.Bitcast (Mir_type.F32, bits)) in
      emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, f))
  | Ssa_op.Decode.F64_to_f64 ->
      emit st (Mir_op.Bitcast (Mir_type.F64, load Mir_width.W64 8L))
  | Ssa_op.Decode.I64 -> load Mir_width.W64 8L
  | Ssa_op.Decode.I64_to_f64 ->
      emit st
        (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, load Mir_width.W64 8L))
  | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
  | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.I16_dequant
  | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I8_dequant ->
      unsupported st op

let encode st op addr (enc : Ssa_op.Encode.t) v =
  with_role st Mir_origin.Role.Encode;
  let store width align x =
    emit_unit st (Mir_op.Store ({ Mir_op.Access.width; addr; align }, x))
  in
  match enc with
  | Ssa_op.Encode.F32_round ->
      let r = emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v)) in
      store Mir_width.W32 4L (emit st (Mir_op.Bitcast (Mir_type.i32, r)))
  | Ssa_op.Encode.I64 -> store Mir_width.W64 8L v
  | Ssa_op.Encode.Bool_nonzero -> unsupported st op

let fbinary = function
  | Expr.Value.Add -> Mir_op.Fbinary.Add
  | Expr.Value.Div -> Mir_op.Fbinary.Div
  | Expr.Value.Mul -> Mir_op.Fbinary.Mul
  | Expr.Value.Sub -> Mir_op.Fbinary.Sub

let compare_op = function
  | Ssa_op.Compare.Eq -> (Mir_op.Icmp.Eq, Mir_op.Fcmp.Eq)
  | Ssa_op.Compare.Lt -> (Mir_op.Icmp.Slt, Mir_op.Fcmp.Lt)

(* One SSA operation; [result] receives its value, if it has one. *)
let instr st (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  let v = value st in
  let result m =
    match List.filter (fun r -> not (is_effect r)) i.Ssa_instr.results with
    | [ r ] -> bind st r m
    | _ -> invalid_arg "Mir_lower: a value result is missing"
  in
  with_role st Mir_origin.Role.Compute;
  match op with
  | Ssa_op.Check_access { buffer; at = Ssa_access.Coord c } ->
      coord_guards st (entry_of st buffer) c
  | Ssa_op.Const c ->
      result
        (const st
           (match c with
           | Ssa_const.F32 x -> Mir_const.f32 x
           | Ssa_const.F64 x -> Mir_const.f64 x
           | Ssa_const.I64 x -> Mir_const.i64 x
           | Ssa_const.Index x -> Mir_const.i32 x
           | Ssa_const.Pred b -> Mir_const.pred b))
  | Ssa_op.Convert (c, a) ->
      result
        (match c with
        | Ssa_op.Convert.F32_to_f64 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v a))
        | Ssa_op.Convert.F64_to_f32 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v a))
        | Ssa_op.Convert.Index_to_f64 ->
            emit st
              (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, sext st (v a)))
        | Ssa_op.Convert.Index_to_i64 -> sext st (v a)
        | Ssa_op.Convert.I64_to_f32 ->
            if mutated st Mutation.Double_rounding then
              let d =
                emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, v a))
              in
              emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, d))
            else emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f32, v a))
        | Ssa_op.Convert.I64_to_f64 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, v a)))
  | Ssa_op.Float_binary (o, a, b) ->
      let a, b =
        match o with
        | (Expr.Value.Div | Expr.Value.Sub)
          when mutated st Mutation.Operand_order ->
            (b, a)
        | _ -> (a, b)
      in
      result (emit st (Mir_op.Fbinary (fbinary o, v a, v b)))
  | Ssa_op.Float_compare (c, a, b) ->
      result (emit st (Mir_op.Fcmp (snd (compare_op c), v a, v b)))
  | Ssa_op.Float_unary (Expr.Value.Sqrt, a) ->
      result (emit st (Mir_op.Funary (Mir_op.Funary.Sqrt, v a)))
  | Ssa_op.Float_unary (Expr.Value.Trunc, a) ->
      result (emit st (Mir_op.Funary (Mir_op.Funary.Trunc, v a)))
  | Ssa_op.I64_arith (o, a, b) ->
      let o =
        match o with
        | Ssa_op.I64_op.Add -> Mir_op.Iarith.Add
        | Ssa_op.I64_op.Mul -> Mir_op.Iarith.Mul
        | Ssa_op.I64_op.Sub -> Mir_op.Iarith.Sub
      in
      result (emit st (Mir_op.Iarith (o, v a, v b)))
  | Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare (c, a, b) ->
      result (icmp st (fst (compare_op c)) (v a) (v b))
  | Ssa_op.Check_gather { raw; extent } ->
      let x = v raw in
      let e = i64k st extent in
      with_role st Mir_origin.Role.Guard;
      let ok =
        pand st
          (icmp st Mir_op.Icmp.Sle (i64k st (Int64.neg extent)) x)
          (icmp st Mir_op.Icmp.Slt x e)
      in
      guard st ~ok Mir_failure.Gather_index_out_of_range (fun () -> [ x; e ])
  | Ssa_op.Float_fma (a, b, c) -> result (emit st (Mir_op.Ffma (v a, v b, v c)))
  | Ssa_op.Float_max (a, b) ->
      result (emit st (Mir_op.Fbinary (Mir_op.Fbinary.Max, v a, v b)))
  | Ssa_op.Float_to_i64 a ->
      let x = v a in
      with_role st Mir_origin.Role.Guard;
      let nan () =
        guard st
          ~ok:(pnot st (fcmp st Mir_op.Fcmp.Unordered x x))
          Mir_failure.I64_from_float_nan
          (fun () -> [])
      in
      let infinite () =
        let inf =
          emit st
            (Mir_op.Pbinary
               ( Mir_op.Pbinary.Or,
                 fcmp st Mir_op.Fcmp.Eq x (f64k st Float.infinity),
                 fcmp st Mir_op.Fcmp.Eq x (f64k st Float.neg_infinity) ))
        in
        guard st ~ok:(pnot st inf) Mir_failure.I64_from_float_infinite
          (fun () -> [])
      in
      let range () =
        let ok =
          pand st
            (fcmp st Mir_op.Fcmp.Le (f64k st (-9223372036854775808.)) x)
            (fcmp st Mir_op.Fcmp.Lt x (f64k st 9223372036854775808.))
        in
        guard st ~ok Mir_failure.I64_from_float_out_of_range (fun () -> [ x ])
      in
      (* NaN, then an infinity, then the range: a NaN or an infinity also
         fails the range test, so this order is the row *)
      if mutated st Mutation.Conversion_order then (
        range ();
        nan ();
        infinite ())
      else (
        nan ();
        infinite ();
        range ());
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Fto_sint x))
  | Ssa_op.I64_div (a, b) ->
      let x = v a and y = v b in
      with_role st Mir_origin.Role.Guard;
      guard st
        ~ok:(pnot st (icmp st Mir_op.Icmp.Eq y (i64k st 0L)))
        Mir_failure.I64_division_by_zero
        (fun () -> []);
      let both =
        pand st
          (icmp st Mir_op.Icmp.Eq x (i64k st Int64.min_int))
          (icmp st Mir_op.Icmp.Eq y (i64k st (-1L)))
      in
      guard st ~ok:(pnot st both) Mir_failure.I64_division_overflow (fun () ->
          []);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Idiv (Mir_op.Idiv.Div_s, x, y)))
  | Ssa_op.Index_floor_div (k, a) | Ssa_op.Index_ceil_div (k, a) ->
      (* by a positive literal: the truncated quotient, adjusted by the
         remainder's sign *)
      let n = sext st (v a) and d = i64k st k in
      let q = emit st (Mir_op.Idiv (Mir_op.Idiv.Div_s, n, d)) in
      let r = emit st (Mir_op.Idiv (Mir_op.Idiv.Rem_s, n, d)) in
      let zero = i64k st 0L and one = i64k st 1L in
      let q =
        match op with
        | Ssa_op.Index_floor_div _ ->
            select st
              (icmp st Mir_op.Icmp.Slt r zero)
              (emit st (Mir_op.Iarith (Mir_op.Iarith.Sub, q, one)))
              q
        | _ ->
            select st
              (icmp st Mir_op.Icmp.Slt zero r)
              (emit st (Mir_op.Iarith (Mir_op.Iarith.Add, q, one)))
              q
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, q)))
  | Ssa_op.Index_of_i64 a ->
      result (emit st (Mir_op.Narrow (Mir_width.W32, v a)))
  | Ssa_op.Pool_better (best, value) ->
      let b = v best and x = v value in
      result
        (emit st
           (Mir_op.Pbinary
              ( Mir_op.Pbinary.Or,
                fcmp st Mir_op.Fcmp.Lt b x,
                fcmp st Mir_op.Fcmp.Unordered x x )))
  | Ssa_op.Index_add (a, b) ->
      let x = sext st (v a) and y = sext st (v b) in
      let sum = emit st (Mir_op.Iarith (Mir_op.Iarith.Add, x, y)) in
      with_role st Mir_origin.Role.Guard;
      guard st ~ok:(in_index_domain st sum)
        (Mir_failure.Index_overflow Mir_failure.Overflow_op.Add) (fun () ->
          [ x; y ]);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Itrunc (Mir_width.W32, sum)))
  | Ssa_op.Index_add_in_domain (a, b) ->
      let sum =
        emit st
          (Mir_op.Iarith (Mir_op.Iarith.Add, sext st (v a), sext st (v b)))
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, sum)))
  | Ssa_op.Index_scale (k, a) ->
      let kk = i64k st k and y = sext st (v a) in
      let prod = emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, kk, y)) in
      with_role st Mir_origin.Role.Guard;
      guard st ~ok:(in_index_domain st prod)
        (Mir_failure.Index_overflow Mir_failure.Overflow_op.Mul) (fun () ->
          [ kk; y ]);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Itrunc (Mir_width.W32, prod)))
  | Ssa_op.Index_scale_in_domain (k, a) ->
      let prod =
        emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, i64k st k, sext st (v a)))
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, prod)))
  | Ssa_op.Index_clamp_low a ->
      let x = v a in
      let zero = i32k st 0L in
      result (select st (icmp st Mir_op.Icmp.Slt x zero) zero x)
  | Ssa_op.Index_max (a, b) ->
      let x = v a and y = v b in
      result (select st (icmp st Mir_op.Icmp.Slt x y) y x)
  | Ssa_op.Index_min (a, b) ->
      let x = v a and y = v b in
      result (select st (icmp st Mir_op.Icmp.Slt x y) x y)
  | Ssa_op.Load { buffer; at; decode = d } ->
      let e = entry_of st buffer in
      if mutated st Mutation.Eager_load then (
        let x = decode st op (address st e at) d in
        (match at with
        | Ssa_access.Coord c -> coord_guards st e c
        | Ssa_access.Flat _ -> ());
        result x)
      else (
        (match at with
        | Ssa_access.Coord c -> coord_guards st e c
        | Ssa_access.Flat _ -> ());
        result (decode st op (address st e at) d))
  | Ssa_op.Load_in_bounds { buffer; at; decode = d } ->
      let e = entry_of st buffer in
      result (decode st op (address st e at) d)
  | Ssa_op.Mark m ->
      emit_unit st (Mir_op.Event (Mir_census_event.of_mark m, 1L))
  | Ssa_op.Mark_lanes { mark; lanes } ->
      emit_unit st
        (Mir_op.Event
           ( Mir_census_event.of_mark mark,
             Int64.of_int (Ssa_type.Lanes.to_int lanes) ))
  | Ssa_op.Pred_not a -> result (emit st (Mir_op.Pnot (v a)))
  | Ssa_op.Pred_or (a, b) ->
      result (emit st (Mir_op.Pbinary (Mir_op.Pbinary.Or, v a, v b)))
  | Ssa_op.Select (p, a, b) -> result (select st (v p) (v a) (v b))
  | Ssa_op.Store { buffer; at; encode = enc; value = x } ->
      let e = entry_of st buffer in
      encode st op (address st e at) enc (v x)
  | Ssa_op.Check_access { at = Ssa_access.Flat _; _ }
  | Ssa_op.Check_local _ | Ssa_op.Check_scan _
  | Ssa_op.Float_unary
      ( ( Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
        | Expr.Value.Sin ),
        _ )
  | Ssa_op.Lanewise _ | Ssa_op.Local_alloc _ | Ssa_op.Local_read _
  | Ssa_op.Local_write _ | Ssa_op.Meter_charge | Ssa_op.Meter_release _
  | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset | Ssa_op.Vec_extract _
  | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _
  | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
      unsupported st op

let data_values vs = List.filter (fun v -> not (is_effect v)) vs

let edge st (e : Ssa_cfg_edge.t) =
  let target = Hashtbl.find st.heads (e.Ssa_cfg_edge.target :> int) in
  let args = List.map (value st) (data_values e.Ssa_cfg_edge.args) in
  let args =
    if not (mutated st Mutation.Sequential_transfer) then args
    else
      (* argument [i] naming the target's parameter [j < i] reads the value
         parameter [j] was just given, as a sequential copy would *)
      let params = B.param target in
      let given = Array.of_list args in
      List.mapi
        (fun i (a : Mir_value.t) ->
          let rec index k = function
            | [] -> None
            | p :: rest ->
                if Mir_value.equal a p then Some k else index (k + 1) rest
          in
          match index 0 params with Some j when j < i -> given.(j) | _ -> a)
        args
  in
  (target, args)

let block st (b : Ssa_cfg_block.t) =
  st.cur <- Hashtbl.find st.heads (b.Ssa_cfg_block.id :> int);
  List.iteri
    (fun k (i : Ssa_instr.t) ->
      st.origin <-
        {
          Mir_origin.cfg =
            Some
              {
                Mir_origin.Cfg_site.block = (b.Ssa_cfg_block.id :> int);
                instr = k;
              };
          output =
            (match i.Ssa_instr.origin with
            | Ssa_origin.Output o -> Some (Expr.Source.create (o :> int))
            | Ssa_origin.Unknown -> None);
          role = Mir_origin.Role.Compute;
          clone = 0;
        };
      instr st i)
    b.Ssa_cfg_block.body;
  match b.Ssa_cfg_block.terminator with
  | Ssa_cfg_terminator.Jump e ->
      let target, args = edge st e in
      B.jump st.cur target args
  | Ssa_cfg_terminator.Branch { cond; then_; else_ } ->
      B.branch st.cur (value st cond) (edge st then_) (edge st else_)
  | Ssa_cfg_terminator.Return _ -> B.return st.cur []

(* Binary32 arithmetic or a fused multiply-add anywhere in the graph. *)
let scan (cfg : Ssa_cfg.t) =
  List.fold_left
    (fun (f32, fma) (b : Ssa_cfg_block.t) ->
      List.fold_left
        (fun (f32, fma) (i : Ssa_instr.t) ->
          let is32 (r : Ssa_value.t) =
            match r.Ssa_value.ty with
            | Ssa_type.Scalar Ssa_type.F32 | Ssa_type.Vec (Ssa_type.F32, _) ->
                true
            | _ -> false
          in
          let rec fused = function
            | Ssa_op.Float_fma _ -> true
            | Ssa_op.Lanewise o -> fused o
            | _ -> false
          in
          (* binary32 arithmetic, not a binary32 value a conversion rounds *)
          let rec arithmetic = function
            | Ssa_op.Float_binary _ | Ssa_op.Float_fma _ | Ssa_op.Float_max _
            | Ssa_op.Float_unary _ ->
                true
            | Ssa_op.Lanewise o -> arithmetic o
            | _ -> false
          in
          ( f32
            || arithmetic i.Ssa_instr.op
               && List.exists is32 i.Ssa_instr.results,
            fma || fused i.Ssa_instr.op ))
        (f32, fma) b.Ssa_cfg_block.body)
    (false, false) cfg.Ssa_cfg.blocks

let program ?mutation ~planning (p : Ssa_program.t) =
  Err.Escape.with_escape @@ fun esc ->
  let refuse r = Err.Escape.throw esc r in
  (match Err.payload (Ssa_verify.check p) with
  | Error (`Invalid_program d) -> refuse (Refusal.Invalid_program d)
  | Ok () -> ());
  let cfg =
    match Ssa_cfg_lower.program p with
    | Ok c -> c
    | Error e -> refuse (Refusal.Cfg e)
  in
  (match Err.payload (Ssa_cfg_verify.check cfg) with
  | Error (`Invalid_cfg d) -> refuse (Refusal.Invalid_cfg d)
  | Ok () -> ());
  let f32, fma = scan cfg in
  let planning =
    match Mir_planning.admit planning ~subject:(subject p) ~contracts:fma with
    | Ok s -> s
    | Error m -> refuse (Refusal.Planning m)
  in
  if f32 && planning.Mir_planning.precision = Mir_planning.Precision.F64 then
    refuse (Refusal.Precision planning.Mir_planning.precision);
  let layout =
    match L.entries cfg.Ssa_cfg.buffers with
    | Ok l -> l
    | Error b -> refuse (Refusal.Buffer_layout b)
  in
  let bld = B.create () in
  let entry = B.new_block bld [] in
  let st =
    {
      esc;
      bld;
      layout;
      values = Hashtbl.create 256;
      heads = Hashtbl.create 32;
      cur = entry;
      origin = Mir_origin.unknown;
      mutation;
    }
  in
  let rpo =
    List.filter_map (Ssa_cfg.find_block cfg) (Ssa_cfg.reverse_postorder cfg)
  in
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      let params = data_values b.Ssa_cfg_block.params in
      let mb =
        B.new_block bld
          (List.map
             (fun (v : Ssa_value.t) -> machine_type st v.Ssa_value.ty)
             params)
      in
      List.iter2 (bind st) params (B.param mb);
      Hashtbl.replace st.heads (b.Ssa_cfg_block.id :> int) mb)
    rpo;
  B.jump entry (Hashtbl.find st.heads (cfg.Ssa_cfg.entry :> int)) [];
  (* blocks in reverse postorder, so every use follows its definition *)
  List.iter (block st) rpo;
  let fn = Mir_id.Func.of_int 0 in
  let f = B.func bld ~id:fn ~name:"kernel" ~entry ~results:[] in
  let mir =
    B.program ~regions:(List.map L.region layout)
      ~views:(List.map L.view layout) ~planning [ f ] ~main:fn
  in
  match Err.payload (Mir_verify.generic mir) with
  | Ok g -> { program = g; layout; planning }
  | Error d -> refuse (Refusal.Invalid_lowering d)
