open Machine_ir
open A64_op
module S = A64_stage.Sel

module Refusal = struct
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Invalid_selection of Mir_diagnostic.t
    | Missing_site of Mir_failure.t
    | Operation of string
    | Vector of Mir_vsplit.Refusal.t
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
    | Vector r -> Mir_vsplit.Refusal.pp fmt r
    | Width t ->
        Fmt.pf fmt "type %a has no aarch64 register in this slice" Mir_type.pp t
end

module Mutation = struct
  type t =
    | Commuted_sub
    | Contiguous_lanes
    | Contract
    | Dropped_half
    | Fcmp_lt_cond
    | Missing_failure_word
    | Pruned_live
    | Signed_compare
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

(* The arrangement a register-wide vector value has. *)
let arr_of st (t : Mir_type.t) =
  match t with
  | Mir_type.Vec (Mir_type.Elem.F64, n) when Mir_type.Lanes.to_int n = 2 ->
      Arr.D2
  | Mir_type.Vec (Mir_type.Elem.F32, n) when Mir_type.Lanes.to_int n = 2 ->
      Arr.S2
  | Mir_type.Vec (Mir_type.Elem.F32, n) when Mir_type.Lanes.to_int n = 4 ->
      Arr.S4
  | _ -> refuse st (Refusal.Width t)

let is_vector (v : Mir_value.t) =
  match v.Mir_value.ty with
  | Mir_type.Vec _ | Mir_type.Mask _ -> true
  | _ -> false

(* [a op c] with a constant operand the form encodes ([ok]): the other operand
   and the constant, which is [c] unless the operation [commutes]. *)
let immediate st ~commutes ~ok a c =
  let imm v =
    match const_of st v with Some k when ok k -> Some k | _ -> None
  in
  match imm c with
  | Some k -> Some (a, k)
  | None when commutes -> Option.map (fun k -> (c, k)) (imm a)
  | None -> None

(* A compare's condition and the NZCV it sets. *)
let condition st (op : Mir_op.t) =
  let ty (v : Mir_value.t) = v.Mir_value.ty in
  match op with
  | Mir_op.Icmp (c, a, b) ->
      let sz = sz_of st (ty a) in
      ( icond st c,
        emit st Mir_type.Flags
          (match immediate st ~commutes:false ~ok:imm12 a b with
          | Some (x, k) -> Cmp_imm (sz, x, k)
          | None -> Cmp (sz, a, b)) )
  | Mir_op.Fcmp (c, a, b) ->
      (fcond st c, emit st Mir_type.Flags (Fcmp (fsz_of st (ty a), a, b)))
  | _ -> invalid_arg "A64_select.condition: not a compare"

(* [base + off] as a pointer. *)
let offset st base off =
  if Int64.equal off 0L then base
  else if imm12 off then emit st Mir_type.Ptr (Add_imm (Sz.X, base, off))
  else emit st Mir_type.Ptr (Add (Sz.X, base, materialize st Mir_type.i64 off))

(* An access address as a base and an immediate: a pointer plus a constant
   the form's offset field takes ([ok]) folds; anything else is its own base. *)
let displaced st (addr : Mir_value.t) ~ok =
  match
    Hashtbl.find_opt st.b.Mir_select.defs
      (Mir_id.Value.to_int addr.Mir_value.id)
  with
  | Some (Mir_op.Ptr_add (base, k)) -> (
      match Mir_select.const_of st.b k with
      | Some c when ok c -> (base, c)
      | _ -> (addr, 0L))
  | _ -> (addr, 0L)

(* [n] ordered forms in a row, chained on the order [o] threads. *)
let chained st (o : Mir_order.t option) n f =
  let o = Option.get o in
  let input = ref o.Mir_order.input in
  for k = 0 to n - 1 do
    let output =
      if k = n - 1 then o.Mir_order.output else fresh st Mir_type.Order
    in
    f k { Mir_order.input = !input; output };
    input := output
  done

(* A register-wide vector operation: Advanced SIMD forms on Q (4S, 2D) or D
   (2S) registers. A strided access is one lane access per lane, in lane
   order. *)
let vector st (i : Mir_op.t Mir_instr.t) =
  let r () =
    match i.Mir_instr.results with
    | [ r ] -> r
    | _ -> invalid_arg "A64_select: a vector result"
  in
  let def op = ignore (emit st ~result:(r ()) (r ()).Mir_value.ty op) in
  let arr (v : Mir_value.t) = arr_of st v.Mir_value.ty in
  let unsupported () =
    refuse st (Refusal.Operation (Mir_op.name i.Mir_instr.op))
  in
  match i.Mir_instr.op with
  | Mir_op.Copy a -> def (Vmov (arr a, a))
  | Mir_op.Fbinary (o, a, b) ->
      let o =
        match o with
        | Mir_op.Fbinary.Add -> Fop.Add
        | Mir_op.Fbinary.Div -> Fop.Div
        | Mir_op.Fbinary.Max -> Fop.Max
        | Mir_op.Fbinary.Mul -> Fop.Mul
        | Mir_op.Fbinary.Sub -> Fop.Sub
      in
      def (Vfbin (o, arr a, a, b))
  | Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, a) when arr a = Arr.S2 ->
      def (Fcvtl a)
  | Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, a) when arr a = Arr.D2 ->
      def (Fcvtn a)
  | Mir_op.Ffma (a, b, c) -> def (Vfmla (arr a, c, a, b))
  | Mir_op.Funary (u, a) ->
      let u =
        match u with
        | Mir_op.Funary.Neg -> Funary.Fneg
        | Mir_op.Funary.Sqrt -> Funary.Fsqrt
        | Mir_op.Funary.Trunc -> Funary.Frintz
      in
      def (Vfunary (u, arr a, a))
  | Mir_op.Vconcat [ a; b ] when arr a = Arr.S2 && arr b = Arr.S2 ->
      if mutated st Mutation.Dropped_half then def (Vwiden a)
      else
        let low = emit st (Arr.ty Arr.S4) (Vwiden a) in
        def (Ins_half (low, b))
  | Mir_op.Vextract (lane, a) ->
      def (Dup_lane (Arr.fsz (arr a), Mir_type.Lane.to_int lane, a))
  | Mir_op.Vinsert (lane, a, x) ->
      def (Ins_lane (Arr.fsz (arr a), Mir_type.Lane.to_int lane, a, x))
  | Mir_op.Vload { Mir_op.Vaccess.addr; stride; _ } ->
      let a = arr (r ()) in
      let fsz = Arr.fsz a and n = Arr.lanes a in
      let size = match fsz with Fsz.D -> 8L | Fsz.S -> 4L in
      if a = Arr.S2 then refuse st (Refusal.Width (r ()).Mir_value.ty);
      if Int64.equal stride size || mutated st Mutation.Contiguous_lanes then
        let base, k = displaced st addr ~ok:(A64_op.vec_offset a) in
        push st ?order:i.Mir_instr.order
          [ r () ]
          (Mir_sel.Op.Machine (Ldr_vec (a, base, k)))
      else if Int64.equal stride 0L then
        push st ?order:i.Mir_instr.order
          [ r () ]
          (Mir_sel.Op.Machine (Ld1r (a, addr)))
      else
        let acc = ref None in
        chained st i.Mir_instr.order n (fun k order ->
            let res = if k = n - 1 then r () else fresh st (Arr.ty a) in
            let at = offset st addr (Int64.mul (Int64.of_int k) stride) in
            push st ~order [ res ]
              (Mir_sel.Op.Machine
                 (match !acc with
                 | None -> Ld1r (a, at)
                 | Some v -> Ld1_lane (fsz, k, v, at)));
            acc := Some res)
  | Mir_op.Vslice (first, count, a)
    when arr a = Arr.S4
         && Mir_type.Lanes.to_int count = 2
         && Mir_type.Lane.to_int first mod 2 = 0 ->
      def (Dup_half (Mir_type.Lane.to_int first / 2, a))
  | Mir_op.Vsplat (_, x) -> def (Dup_elem (arr (r ()), x))
  | Mir_op.Vstore ({ Mir_op.Vaccess.addr; stride; _ }, v) ->
      let a = arr v in
      let fsz = Arr.fsz a and n = Arr.lanes a in
      let size = match fsz with Fsz.D -> 8L | Fsz.S -> 4L in
      if a = Arr.S2 then refuse st (Refusal.Width v.Mir_value.ty);
      if Int64.equal stride size then
        let base, k = displaced st addr ~ok:(A64_op.vec_offset a) in
        push st ?order:i.Mir_instr.order []
          (Mir_sel.Op.Machine (Str_vec (a, base, k, v)))
      else
        chained st i.Mir_instr.order n (fun k order ->
            let at = offset st addr (Int64.mul (Int64.of_int k) stride) in
            push st ~order [] (Mir_sel.Op.Machine (St1_lane (fsz, k, v, at))))
  | _ -> unsupported ()

let instr st (i : Mir_op.t Mir_instr.t) =
  st.b.Mir_select.origin <- i.Mir_instr.origin;
  if List.exists is_vector (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results)
  then vector st i
  else
    let result = match i.Mir_instr.results with [ r ] -> Some r | _ -> None in
    let r () =
      match result with
      | Some r -> r
      | None -> invalid_arg "A64_select: a result"
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
    | Mir_op.Fcmp _ | Mir_op.Icmp _ ->
        let cond, f = condition st i.Mir_instr.op in
        flags_then st ~result:(r ()) ~cond f
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
        let logic o =
          match
            immediate st ~commutes:true
              ~ok:(bitmask_immediate ~bits:(Sz.bits sz))
              a b
          with
          | Some (x, k) -> def (Logic_imm (o, sz, x, k))
          | None -> def (Logic (o, sz, a, b))
        in
        match o with
        | Mir_op.Iarith.Add -> (
            match immediate st ~commutes:true ~ok:imm12 a b with
            | Some (x, k) -> def (Add_imm (sz, x, k))
            | None -> def (Add (sz, a, b)))
        | Mir_op.Iarith.And -> logic Logic.And
        | Mir_op.Iarith.Mul -> (
            (* by a power of two: a left shift *)
            let log2 k =
              List.find_opt
                (fun n -> Int64.equal k (Int64.shift_left 1L n))
                (List.init (Sz.bits sz) Fun.id)
            in
            match
              immediate st ~commutes:true
                ~ok:(fun k -> Option.is_some (log2 k))
                a b
            with
            | Some (x, k) ->
                def (Shift_imm (Shift.Lsl, sz, x, Option.get (log2 k)))
            | None -> def (Mul (sz, a, b)))
        | Mir_op.Iarith.Or -> logic Logic.Orr
        | Mir_op.Iarith.Shl -> shift Shift.Lsl
        | Mir_op.Iarith.Shr_s -> shift Shift.Asr
        | Mir_op.Iarith.Shr_u -> shift Shift.Lsr
        | Mir_op.Iarith.Sub -> (
            match
              immediate st
                ~commutes:(mutated st Mutation.Commuted_sub)
                ~ok:imm12 a b
            with
            | Some (x, k) -> def (Sub_imm (sz, x, k))
            | None -> def (Sub (sz, a, b)))
        | Mir_op.Iarith.Xor -> logic Logic.Eor)
    | Mir_op.Idiv (Mir_op.Idiv.Div_s, a, b) ->
        def (Sdiv (sz_of st (ty a), a, b))
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
      when Mir_type.equal (ty a) Mir_type.i8
           || Mir_type.equal (ty a) Mir_type.i16 ->
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
        let m = msz_of width in
        let base, k =
          displaced st addr ~ok:(A64.frame_offset_ok ~bytes:(Msz.bytes m))
        in
        push st ?order:i.Mir_instr.order
          [ r () ]
          (Mir_sel.Op.Machine (Ldr (m, base, k)))
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
    | Mir_op.Ptr_add (p, d) -> (
        match const_of st d with
        | Some k when imm12 k -> def (Add_imm (Sz.X, p, k))
        | _ -> def (Add (Sz.X, p, d)))
    | Mir_op.Select (p, a, b) -> (
        let f = emit st Mir_type.Flags (Cmp_imm (Sz.W, p, 0L)) in
        match ty a with
        | Mir_type.F32 | Mir_type.F64 ->
            def (Fcsel (fsz_of st (ty a), Cond.Ne, f, a, b))
        | t -> def (Csel (sz_of st t, Cond.Ne, f, a, b)))
    | Mir_op.Store ({ Mir_op.Access.width; addr; _ }, v) ->
        let m = msz_of width in
        let base, k =
          displaced st addr ~ok:(A64.frame_offset_ok ~bytes:(Msz.bytes m))
        in
        push st ?order:i.Mir_instr.order []
          (Mir_sel.Op.Machine (Str (m, base, k, v)))
    | Mir_op.Undef v -> push st ?order:i.Mir_instr.order [] (Mir_sel.Op.Undef v)
    | Mir_op.Vconcat _ | Mir_op.Vextract _ | Mir_op.Vinsert _ | Mir_op.Vload _
    | Mir_op.Vslice _ | Mir_op.Vsplat _ | Mir_op.Vstore _ ->
        unsupported ()

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

(* A vector is register-wide by now; a mask is never kept in a register. *)
let scalar st vs =
  List.iter
    (fun (v : Mir_value.t) ->
      match v.Mir_value.ty with
      | Mir_type.Mask _ -> refuse st (Refusal.Width v.Mir_value.ty)
      | Mir_type.Vec _ -> ignore (arr_of st v.Mir_value.ty)
      | _ -> ())
    vs

let block st (blk : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
  scalar st blk.Mir_block.params;
  List.iter
    (fun (i : Mir_op.t Mir_instr.t) ->
      scalar st (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results))
    blk.Mir_block.body;
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
        (* a compare of this block is compared again at the branch, and its
           CSET pruned when nothing else reads it *)
        let test =
          match
            List.find_opt
              (fun (i : Mir_op.t Mir_instr.t) ->
                List.exists (Mir_value.equal cond) i.Mir_instr.results)
              body
          with
          | Some ({ Mir_instr.op = Mir_op.Icmp _ | Mir_op.Fcmp _; _ } as i) ->
              let c, f = condition st i.Mir_instr.op in
              B_cond (c, f)
          | _ -> Cbnz (Sz.W, cond)
        in
        Mir_sel.Terminator.Branch
          { test; then_ = ord_edge st then_; else_ = ord_edge st else_ }
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
  Err.Escape.with_escape @@ fun esc ->
  let g =
    match Err.payload (Mir_vsplit.program ~register_bytes:16L g) with
    | Ok g -> g
    | Error r -> Err.Escape.throw esc (Refusal.Vector r)
  in
  let p =
    Mir_verify.Generic.program (Mir_narrow.program (Mir_offsets.program g))
  in
  let fallible = Mir_select.fallibility p in
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
         (S.prune
            ~ignore_terminators:(mutation = Some Mutation.Pruned_live)
            {
              S.features = [ Mir_target.Feature.Fp ];
              program = Mir_select.program p funcs;
            }))
  with
  | Ok v -> { selected = v; record = Mir_select.record_view }
  | Error d -> Err.Escape.throw esc (Refusal.Invalid_selection d)
