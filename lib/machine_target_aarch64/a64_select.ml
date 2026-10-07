open Machine_ir
open A64_op
module S = A64_stage.Sel

module Refusal = struct
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Invalid_selection of Mir_diagnostic.t
    | Missing_site of Mir_failure.t
    | Operation of string
    | Width of Mir_type.t

  let pp fmt = function
    | Fallible_with_results c ->
        Fmt.pf fmt
          "a call to %a that may fail, from or with results this slice cannot \
           leave undefined"
          Mir_op.Callee.pp c
    | Invalid_selection d ->
        Fmt.pf fmt "selection defect: %a" Mir_diagnostic.pp d
    | Missing_site f -> Fmt.pf fmt "no site-table entry for %a" Mir_failure.pp f
    | Operation o -> Fmt.pf fmt "%s is not selected for aarch64" o
    | Width t ->
        Fmt.pf fmt "type %a has no aarch64 register in this slice" Mir_type.pp t
end

module Mutation = struct
  type t = Contract | Fcmp_lt_cond | Missing_failure_word | Signed_compare
end

type result = { selected : S.Verified.t; record : Mir_id.View.t }

type st = {
  b : (A64.op, A64.test) Mir_select.t;
  esc : Refusal.t Err.Escape.t;
  mutation : Mutation.t option;
  sites : Mir_failure.Site_entry.t array;
  unlisted : Mir_failure.Unlisted.t;
  fallible : Mir_op.Callee.t -> bool;
}

let refuse st r = Err.Escape.throw st.esc r
let mutated st m = st.mutation = Some m
let fresh st ty = Mir_select.fresh st.b ty
let push st ?order results op = Mir_select.push st.b ?order results op
let emit st ?result ty op = Mir_select.emit st.b ?result ty op
let ord st v = Mir_select.ord st.b v
let ord_edge st e = Mir_select.ord_edge st.b e
let const_of st v = Mir_select.const_of st.b v

let sz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Sz.X
  | Mir_type.Int Mir_width.W32 | Mir_type.Pred -> Sz.W
  | _ -> refuse st (Refusal.Width t)

let fsz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.F64 -> Fsz.D
  | Mir_type.F32 -> Fsz.S
  | _ -> refuse st (Refusal.Width t)

(* [bits] of an i32 or i64 value: one MOVZ or MOVN when one halfword differs
   from the background, else MOVZ then MOVK per remaining halfword. *)
let materialize st ?result (t : Mir_type.t) bits =
  let n = match t with Mir_type.Int Mir_width.W64 -> 4 | _ -> 2 in
  let half v k =
    Int64.to_int (Int64.logand (Int64.shift_right_logical v (16 * k)) 0xFFFFL)
  in
  let mask = if n = 4 then -1L else 0xFFFF_FFFFL in
  let bits = Int64.logand bits mask in
  let inv = Int64.logand (Int64.lognot bits) mask in
  let nonzero v = List.filter (fun k -> half v k <> 0) (List.init n Fun.id) in
  let sz = if n = 4 then Sz.X else Sz.W in
  match (nonzero bits, nonzero inv) with
  | ([] | [ _ ]), _ ->
      let k = match nonzero bits with [ k ] -> k | _ -> 0 in
      emit st ?result t (Movz (t, half bits k, 16 * k))
  | _, ([] | [ _ ]) ->
      let k = match nonzero inv with [ k ] -> k | _ -> 0 in
      emit st ?result t (Movn (sz, half inv k, 16 * k))
  | first :: rest, _ ->
      let v = emit st t (Movz (t, half bits first, 16 * first)) in
      let rec go v = function
        | [] -> v
        | [ k ] -> emit st ?result t (Movk (sz, v, half bits k, 16 * k))
        | k :: more -> go (emit st t (Movk (sz, v, half bits k, 16 * k))) more
      in
      go v rest

let const st ?result (c : Mir_const.t) =
  match c.Mir_const.ty with
  | Mir_type.Int Mir_width.W64 | Mir_type.Int Mir_width.W32 ->
      materialize st ?result c.Mir_const.ty c.Mir_const.bits
  | Mir_type.Pred | Mir_type.Int Mir_width.W8 ->
      emit st ?result c.Mir_const.ty
        (Movz (c.Mir_const.ty, Int64.to_int c.Mir_const.bits, 0))
  | Mir_type.F64 ->
      let g = materialize st Mir_type.i64 c.Mir_const.bits in
      emit st ?result Mir_type.F64 (Fmov_from_gpr (Fsz.D, g))
  | Mir_type.F32 ->
      let g = materialize st Mir_type.i32 c.Mir_const.bits in
      emit st ?result Mir_type.F32 (Fmov_from_gpr (Fsz.S, g))
  | t -> refuse st (Refusal.Width t)

let flags_then st ~result ~cond flags =
  ignore (emit st ~result Mir_type.Pred (Cset (cond, flags)))

let icond st (c : Mir_op.Icmp.t) =
  match c with
  | Mir_op.Icmp.Eq -> Cond.Eq
  | Mir_op.Icmp.Ne -> Cond.Ne
  | Mir_op.Icmp.Sle -> Cond.Le
  | Mir_op.Icmp.Slt -> Cond.Lt
  | Mir_op.Icmp.Ule ->
      if mutated st Mutation.Signed_compare then Cond.Le else Cond.Ls
  | Mir_op.Icmp.Ult ->
      if mutated st Mutation.Signed_compare then Cond.Lt else Cond.Lo

(* FCMP leaves N set only for an ordered less-than; [lt] would also be true
   when unordered. *)
let fcond st (c : Mir_op.Fcmp.t) =
  match c with
  | Mir_op.Fcmp.Eq -> Cond.Eq
  | Mir_op.Fcmp.Le -> Cond.Ls
  | Mir_op.Fcmp.Lt ->
      if mutated st Mutation.Fcmp_lt_cond then Cond.Lt else Cond.Mi
  | Mir_op.Fcmp.Unordered -> Cond.Vs

let msz_of (w : Mir_width.t) =
  match w with
  | Mir_width.W8 -> Msz.B
  | Mir_width.W16 -> Msz.H
  | Mir_width.W32 -> Msz.W
  | Mir_width.W64 -> Msz.X

let instr st (i : Mir_op.t Mir_instr.t) =
  st.b.Mir_select.origin <- i.Mir_instr.origin;
  let result = match i.Mir_instr.results with [ r ] -> Some r | _ -> None in
  let r () =
    match result with Some r -> r | None -> invalid_arg "A64_select: a result"
  in
  let def op = ignore (emit st ~result:(r ()) (r ()).Mir_value.ty op) in
  let unsupported () =
    refuse st (Refusal.Operation (Mir_op.name i.Mir_instr.op))
  in
  let ty (v : Mir_value.t) = v.Mir_value.ty in
  match i.Mir_instr.op with
  | Mir_op.Addr view ->
      let page = emit st Mir_type.Ptr (Adrp view) in
      def (Add_lo12 (page, view))
  | Mir_op.Bitcast (t, a) -> (
      match (ty a, t) with
      | Mir_type.Int Mir_width.W32, Mir_type.F32 ->
          def (Fmov_from_gpr (Fsz.S, a))
      | Mir_type.Int Mir_width.W64, Mir_type.F64 ->
          def (Fmov_from_gpr (Fsz.D, a))
      | Mir_type.F32, _ -> def (Fmov_to_gpr (Fsz.S, a))
      | _ -> def (Fmov_to_gpr (Fsz.D, a)))
  | Mir_op.Call (callee, args) ->
      let results = i.Mir_instr.results in
      let status = fresh st Mir_type.i32 in
      push st ?order:i.Mir_instr.order (results @ [ status ])
        (Mir_sel.Op.Machine
           (Bl
              {
                callee;
                args;
                results =
                  List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) results;
              }));
      if st.fallible callee then (
        (* status first: a failure propagates with its record untouched *)
        if results <> [] || st.b.Mir_select.results <> [] then
          refuse st (Refusal.Fallible_with_results callee);
        Mir_select.split_on_status st.b
          ~nonzero:(fun s -> Cbnz (Sz.W, s))
          ~status ~after:(Option.get i.Mir_instr.order).Mir_order.output)
  | Mir_op.Const c -> ignore (const st ~result:(r ()) c)
  | Mir_op.Copy a -> (
      match ty a with
      | Mir_type.F32 | Mir_type.F64 -> def (Fmov (fsz_of st (ty a), a))
      | t -> def (Mov (sz_of st t, a)))
  | Mir_op.Event (e, n) ->
      push st ?order:i.Mir_instr.order [] (Mir_sel.Op.Event (e, n))
  | Mir_op.Fbinary (o, a, b) ->
      let o =
        match o with
        | Mir_op.Fbinary.Add -> Fop.Add
        | Mir_op.Fbinary.Div -> Fop.Div
        | Mir_op.Fbinary.Max -> Fop.Max
        | Mir_op.Fbinary.Mul -> Fop.Mul
        | Mir_op.Fbinary.Sub -> Fop.Sub
      in
      def (Fbin (o, fsz_of st (ty a), a, b))
  | Mir_op.Fcmp (c, a, b) ->
      let f = emit st Mir_type.Flags (Fcmp (fsz_of st (ty a), a, b)) in
      flags_then st ~result:(r ()) ~cond:(fcond st c) f
  | Mir_op.Fconvert (c, a) ->
      def
        (match c with
        | Mir_op.Fconvert.F32_to_f64 -> Fcvt (Fsz.D, a)
        | Mir_op.Fconvert.F64_to_f32 -> Fcvt (Fsz.S, a)
        | Mir_op.Fconvert.S64_to_f32 -> Scvtf (Fsz.S, a)
        | Mir_op.Fconvert.S64_to_f64 -> Scvtf (Fsz.D, a))
  | Mir_op.Ffma (a, b, c) -> def (Fmadd (fsz_of st (ty a), a, b, c))
  | Mir_op.Fto_sint a -> def (Fcvtzs (Fsz.D, a))
  | Mir_op.Funary (u, a) ->
      def
        (Funary
           ( (match u with
             | Mir_op.Funary.Neg -> Funary.Fneg
             | Mir_op.Funary.Sqrt -> Funary.Fsqrt
             | Mir_op.Funary.Trunc -> Funary.Frintz),
             fsz_of st (ty a),
             a ))
  | Mir_op.Iarith (o, a, b) -> (
      let sz = sz_of st (ty a) in
      let shift k =
        match const_of st b with
        | Some c -> def (Shift_imm (k, sz, a, Int64.to_int c))
        | None -> unsupported ()
      in
      match o with
      | Mir_op.Iarith.Add -> def (Add (sz, a, b))
      | Mir_op.Iarith.And -> def (Logic (Logic.And, sz, a, b))
      | Mir_op.Iarith.Mul -> def (Mul (sz, a, b))
      | Mir_op.Iarith.Or -> def (Logic (Logic.Orr, sz, a, b))
      | Mir_op.Iarith.Shl -> shift Shift.Lsl
      | Mir_op.Iarith.Shr_s -> shift Shift.Asr
      | Mir_op.Iarith.Shr_u -> shift Shift.Lsr
      | Mir_op.Iarith.Sub -> def (Sub (sz, a, b))
      | Mir_op.Iarith.Xor -> def (Logic (Logic.Eor, sz, a, b)))
  | Mir_op.Icmp (c, a, b) ->
      let f = emit st Mir_type.Flags (Cmp (sz_of st (ty a), a, b)) in
      flags_then st ~result:(r ()) ~cond:(icond st c) f
  | Mir_op.Idiv (Mir_op.Idiv.Div_s, a, b) -> def (Sdiv (sz_of st (ty a), a, b))
  | Mir_op.Idiv (Mir_op.Idiv.Rem_s, a, b) ->
      let sz = sz_of st (ty a) in
      let q = emit st (ty a) (Sdiv (sz, a, b)) in
      def (Msub (sz, q, b, a))
  | Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Sxtw a)
  | Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W64, a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Uxtw a)
  | Mir_op.Iext (k, Mir_width.W32, a)
    when Mir_type.equal (ty a) Mir_type.i8 || Mir_type.equal (ty a) Mir_type.i16
    ->
      let from = match ty a with Mir_type.Int w -> w | _ -> Mir_width.W8 in
      def (Ext { signed = k = Mir_op.Iext.Sext; from; src = a })
  | Mir_op.Iext _ -> unsupported ()
  | (Mir_op.Itrunc (Mir_width.W32, a) | Mir_op.Narrow (Mir_width.W32, a))
    when Mir_type.equal (ty a) Mir_type.i64 ->
      def (Wtrunc a)
  | Mir_op.Itrunc (((Mir_width.W8 | Mir_width.W16) as w), a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Trunc (w, a))
  | Mir_op.Itrunc _ | Mir_op.Narrow _ -> unsupported ()
  | Mir_op.Load { Mir_op.Access.width; addr; _ } ->
      push st ?order:i.Mir_instr.order
        [ r () ]
        (Mir_sel.Op.Machine (Ldr (msz_of width, addr, 0L)))
  | Mir_op.Pbinary (o, a, b) ->
      def
        (Logic
           ( (match o with
             | Mir_op.Pbinary.And -> Logic.And
             | Mir_op.Pbinary.Or -> Logic.Orr
             | Mir_op.Pbinary.Xor -> Logic.Eor),
             Sz.W,
             a,
             b ))
  | Mir_op.Pnot a -> def (Logic_imm (Logic.Eor, Sz.W, a, 1L))
  | Mir_op.Ptr_add (p, d) -> def (Add (Sz.X, p, d))
  | Mir_op.Select (p, a, b) -> (
      let f = emit st Mir_type.Flags (Cmp_imm (Sz.W, p, 0L)) in
      match ty a with
      | Mir_type.F32 | Mir_type.F64 ->
          def (Fcsel (fsz_of st (ty a), Cond.Ne, f, a, b))
      | t -> def (Csel (sz_of st t, Cond.Ne, f, a, b)))
  | Mir_op.Store ({ Mir_op.Access.width; addr; _ }, v) ->
      push st ?order:i.Mir_instr.order []
        (Mir_sel.Op.Machine (Str (msz_of width, addr, 0L, v)))
  | Mir_op.Undef v -> push st ?order:i.Mir_instr.order [] (Mir_sel.Op.Undef v)

(* The failure record stores, then status 1. *)
let fail st (f : Mir_fail.t) =
  st.b.Mir_select.origin <- f.Mir_fail.origin;
  let base () =
    let page = emit st Mir_type.Ptr (Adrp Mir_select.record_view) in
    emit st Mir_type.Ptr (Add_lo12 (page, Mir_select.record_view))
  in
  match
    Mir_select.store_record st.b f ~unlisted:st.unlisted ~sites:st.sites ~base
      ~const:(fun ty bits -> materialize st ty bits)
      ~float_bits:(fun v -> emit st Mir_type.i64 (Fmov_to_gpr (Fsz.D, v)))
      ~store:(fun base off v ~wide ->
        Str ((if wide then Msz.X else Msz.W), base, off, v))
      ~one:(fun () -> emit st Mir_type.i32 (Movz (Mir_type.i32, 1, 0)))
      ?skip:
        (if mutated st Mutation.Missing_failure_word then Some `Last else None)
      ()
  with
  | Some term -> term
  | None -> refuse st (Refusal.Missing_site f.Mir_fail.failure)

let block st (blk : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
  Mir_select.start st.b blk.Mir_block.id blk.Mir_block.params
    blk.Mir_block.order;
  let body =
    if mutated st Mutation.Contract then Mir_select.contract blk.Mir_block.body
    else blk.Mir_block.body
  in
  List.iter (instr st) body;
  (* a terminator's expansion — a failure's record, a return's status —
     belongs to the block's last operation: for a failure block, the payload
     of the guard that failed *)
  st.b.Mir_select.origin <-
    (match List.rev body with
    | (last : Mir_op.t Mir_instr.t) :: _ -> last.Mir_instr.origin
    | [] -> Mir_origin.unknown);
  let terminator =
    match blk.Mir_block.terminator with
    | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
        Mir_sel.Terminator.Branch
          {
            test = Cbnz (Sz.W, cond);
            then_ = ord_edge st then_;
            else_ = ord_edge st else_;
          }
    | Mir_terminator.Jump e -> Mir_sel.Terminator.Jump (ord_edge st e)
    | Mir_terminator.Return { Mir_return.values; order } ->
        let status = emit st Mir_type.i32 (Movz (Mir_type.i32, 0, 0)) in
        Mir_sel.Terminator.Return
          { Mir_return.values = values @ [ status ]; order = ord st order }
    | Mir_terminator.Fail f -> fail st f
  in
  Mir_select.finish st.b terminator

let program ?mutation ?(sites = [||]) ?(unlisted = Mir_failure.Unlisted.Refused)
    (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  let fallible = Mir_select.fallibility p in
  Err.Escape.with_escape @@ fun esc ->
  let funcs =
    List.map
      (fun (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) ->
        let st =
          { b = Mir_select.create f; esc; mutation; sites; unlisted; fallible }
        in
        List.iter (block st) f.Mir_func.blocks;
        {
          Mir_func.id = f.Mir_func.id;
          name = f.Mir_func.name;
          entry = f.Mir_func.entry;
          results = f.Mir_func.results @ [ Mir_type.i32 ];
          blocks = List.rev st.b.Mir_select.finished;
        })
      p.Mir_program.funcs
  in
  match
    Err.payload
      (S.verify
         {
           S.features = [ Mir_target.Feature.Fp ];
           program = Mir_select.program p funcs;
         })
  with
  | Ok v -> { selected = v; record = Mir_select.record_view }
  | Error d -> Err.Escape.throw esc (Refusal.Invalid_selection d)
