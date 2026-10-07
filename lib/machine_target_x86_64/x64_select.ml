open Machine_ir
open X64_op
module S = X64_stage.Sel

module Refusal = struct
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Feature of Mir_target.Feature.t
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
    | Feature f ->
        Fmt.pf fmt "needs %s, which the program's features do not include"
          (Mir_target.Feature.name f)
    | Invalid_selection d ->
        Fmt.pf fmt "selection defect: %a" Mir_diagnostic.pp d
    | Missing_site f -> Fmt.pf fmt "no site-table entry for %a" Mir_failure.pp f
    | Operation o -> Fmt.pf fmt "%s is not selected for x86_64" o
    | Width t ->
        Fmt.pf fmt "type %a has no x86_64 register in this slice" Mir_type.pp t
end

module Mutation = struct
  type t =
    | Contract
    | Division_swap
    | Max_no_nan
    | Missing_failure_word
    | No_parity
    | Scaled_address
    | Signed_compare
end

type result = { selected : S.Verified.t; record : Mir_id.View.t }

type st = {
  b : (X64.op, X64.test) Mir_select.t;
  esc : Refusal.t Err.Escape.t;
  mutation : Mutation.t option;
  sites : Mir_failure.Site_entry.t array;
  unlisted : Mir_failure.Unlisted.t;
  fallible : Mir_op.Callee.t -> bool;
  features : Mir_target.Feature.t list;
}

let refuse st r = Err.Escape.throw st.esc r
let mutated st m = st.mutation = Some m
let fresh st ty = Mir_select.fresh st.b ty
let push st ?order results op = Mir_select.push st.b ?order results op
let emit st ?result ty op = Mir_select.emit st.b ?result ty op
let ord st v = Mir_select.ord st.b v
let ord_edge st e = Mir_select.ord_edge st.b e

let sz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Sz.Q
  | Mir_type.Int Mir_width.W32 | Mir_type.Pred -> Sz.L
  | _ -> refuse st (Refusal.Width t)

let fsz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.F64 -> Fsz.D
  | Mir_type.F32 -> Fsz.S
  | _ -> refuse st (Refusal.Width t)

let need_feature st f =
  if not (List.mem f st.features) then refuse st (Refusal.Feature f)

let const st ?result (c : Mir_const.t) =
  match c.Mir_const.ty with
  | Mir_type.Int _ | Mir_type.Pred ->
      emit st ?result c.Mir_const.ty
        (Mov_imm (c.Mir_const.ty, c.Mir_const.bits))
  | Mir_type.F64 ->
      let g = emit st Mir_type.i64 (Mov_imm (Mir_type.i64, c.Mir_const.bits)) in
      emit st ?result Mir_type.F64 (Movq_from_gpr (Fsz.D, g))
  | Mir_type.F32 ->
      let g = emit st Mir_type.i32 (Mov_imm (Mir_type.i32, c.Mir_const.bits)) in
      emit st ?result Mir_type.F32 (Movq_from_gpr (Fsz.S, g))
  | t -> refuse st (Refusal.Width t)

let icond st (c : Mir_op.Icmp.t) =
  match c with
  | Mir_op.Icmp.Eq -> Cond.E
  | Mir_op.Icmp.Ne -> Cond.Ne
  | Mir_op.Icmp.Sle -> Cond.Le
  | Mir_op.Icmp.Slt -> Cond.L
  | Mir_op.Icmp.Ule ->
      if mutated st Mutation.Signed_compare then Cond.Le else Cond.Be
  | Mir_op.Icmp.Ult ->
      if mutated st Mutation.Signed_compare then Cond.L else Cond.B

let msz_of (w : Mir_width.t) =
  match w with
  | Mir_width.W8 -> Msz.B
  | Mir_width.W16 -> Msz.H
  | Mir_width.W32 -> Msz.L
  | Mir_width.W64 -> Msz.Q

(* An address operand: [base + index * scale] when the generic address is a
   pointer plus an index times 1, 2, 4 or 8. *)
let address st (a : Mir_value.t) =
  let def v =
    Hashtbl.find_opt st.b.Mir_select.defs (Mir_id.Value.to_int v.Mir_value.id)
  in
  match def a with
  | Some (Mir_op.Ptr_add (base, off)) -> (
      match def off with
      | Some (Mir_op.Iarith (Mir_op.Iarith.Mul, x, k)) -> (
          match Mir_select.const_of st.b k with
          | Some s when List.mem s [ 1L; 2L; 4L; 8L ] ->
              let s =
                if mutated st Mutation.Scaled_address then Int64.mul 2L s else s
              in
              { Addr.base; index = Some (x, s); disp = 0L }
          | _ -> { Addr.base = a; index = None; disp = 0L })
      | _ -> { Addr.base = a; index = None; disp = 0L })
  | _ -> { Addr.base = a; index = None; disp = 0L }

(* IEEE maximum from MAXSD, which returns its source operand for a NaN or two
   zeros: two equal operands give their AND (+0 over -0), and a NaN operand
   gives their sum (a NaN). *)
let fmax st ~result fsz a c =
  let t = match fsz with Fsz.D -> Mir_type.F64 | Fsz.S -> Mir_type.F32 in
  let m = emit st t (Fbin (Fop.Max, fsz, a, c)) in
  let e = emit st t (Cmps (Cmp_pred.Eq, fsz, a, c)) in
  let z = emit st t (Flogic (Flogic.And, fsz, a, c)) in
  let t1 = emit st t (Flogic (Flogic.And, fsz, e, z)) in
  let t2 = emit st t (Flogic (Flogic.Andn, fsz, e, m)) in
  if mutated st Mutation.Max_no_nan then
    ignore (emit st ~result t (Flogic (Flogic.Or, fsz, t1, t2)))
  else
    let r0 = emit st t (Flogic (Flogic.Or, fsz, t1, t2)) in
    let u = emit st t (Cmps (Cmp_pred.Unord, fsz, a, c)) in
    let s = emit st t (Fbin (Fop.Add, fsz, a, c)) in
    let t3 = emit st t (Flogic (Flogic.And, fsz, u, s)) in
    let t4 = emit st t (Flogic (Flogic.Andn, fsz, u, r0)) in
    ignore (emit st ~result t (Flogic (Flogic.Or, fsz, t3, t4)))

(* [p ? a : c] for floats: an all-ones or zero mask chosen by CMOV, moved to
   the XMM bank, then AND, ANDN and OR. *)
let fselect st ~result fsz p a c =
  let t = match fsz with Fsz.D -> Mir_type.F64 | Fsz.S -> Mir_type.F32 in
  let it, sz =
    match fsz with
    | Fsz.D -> (Mir_type.i64, Sz.Q)
    | Fsz.S -> (Mir_type.i32, Sz.L)
  in
  let f = emit st Mir_type.Flags (Test (Sz.L, p, p)) in
  let zero = emit st it (Mov_imm (it, 0L))
  and ones =
    emit st it
      (Mov_imm
         ( it,
           Mir_width.mask
             (match fsz with Fsz.D -> Mir_width.W64 | Fsz.S -> Mir_width.W32) ))
  in
  let mask = emit st it (Cmov (sz, Cond.Ne, f, ones, zero)) in
  let m = emit st t (Movq_from_gpr (fsz, mask)) in
  let x = emit st t (Flogic (Flogic.And, fsz, m, a)) in
  let y = emit st t (Flogic (Flogic.Andn, fsz, m, c)) in
  ignore (emit st ~result t (Flogic (Flogic.Or, fsz, x, y)))

let instr st (i : Mir_op.t Mir_instr.t) =
  st.b.Mir_select.origin <- i.Mir_instr.origin;
  let result = match i.Mir_instr.results with [ r ] -> Some r | _ -> None in
  let r () =
    match result with Some r -> r | None -> invalid_arg "X64_select: a result"
  in
  let def op = ignore (emit st ~result:(r ()) (r ()).Mir_value.ty op) in
  let unsupported () =
    refuse st (Refusal.Operation (Mir_op.name i.Mir_instr.op))
  in
  let ty (v : Mir_value.t) = v.Mir_value.ty in
  let setcc c f = def (Setcc_zx (c, f)) in
  match i.Mir_instr.op with
  | Mir_op.Addr view -> def (Lea_view view)
  | Mir_op.Bitcast (t, a) -> (
      match (ty a, t) with
      | Mir_type.Int Mir_width.W32, Mir_type.F32 ->
          def (Movq_from_gpr (Fsz.S, a))
      | Mir_type.Int Mir_width.W64, Mir_type.F64 ->
          def (Movq_from_gpr (Fsz.D, a))
      | Mir_type.F32, _ -> def (Movq_to_gpr (Fsz.S, a))
      | _ -> def (Movq_to_gpr (Fsz.D, a)))
  | Mir_op.Call (callee, args) ->
      let results = i.Mir_instr.results in
      let status = fresh st Mir_type.i32 in
      push st ?order:i.Mir_instr.order (results @ [ status ])
        (Mir_sel.Op.Machine
           (Call
              {
                callee;
                args;
                results =
                  List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) results;
              }));
      if st.fallible callee then (
        if results <> [] || st.b.Mir_select.results <> [] then
          refuse st (Refusal.Fallible_with_results callee);
        let f = emit st Mir_type.Flags (Test (Sz.L, status, status)) in
        Mir_select.split_on_status st.b
          ~nonzero:(fun _ -> Jcc (Cond.Ne, f))
          ~status ~after:(Option.get i.Mir_instr.order).Mir_order.output)
  | Mir_op.Const c -> ignore (const st ~result:(r ()) c)
  | Mir_op.Copy a -> (
      match ty a with
      | Mir_type.F32 | Mir_type.F64 -> def (Movap a)
      | t -> def (Mov (sz_of st t, a)))
  | Mir_op.Event (e, n) ->
      push st ?order:i.Mir_instr.order [] (Mir_sel.Op.Event (e, n))
  | Mir_op.Fbinary (Mir_op.Fbinary.Max, a, c) ->
      fmax st ~result:(r ()) (fsz_of st (ty a)) a c
  | Mir_op.Fbinary (o, a, c) ->
      let o =
        match o with
        | Mir_op.Fbinary.Add -> Fop.Add
        | Mir_op.Fbinary.Div -> Fop.Div
        | Mir_op.Fbinary.Mul -> Fop.Mul
        | Mir_op.Fbinary.Sub -> Fop.Sub
        | Mir_op.Fbinary.Max -> Fop.Max
      in
      def (Fbin (o, fsz_of st (ty a), a, c))
  | Mir_op.Fcmp (c, a, b) -> (
      let fsz = fsz_of st (ty a) in
      match c with
      | Mir_op.Fcmp.Eq ->
          (* equal and ordered: ZF set and PF clear *)
          let f = emit st Mir_type.Flags (Ucomis (fsz, a, b)) in
          if mutated st Mutation.No_parity then setcc Cond.E f
          else
            let e = emit st Mir_type.Pred (Setcc_zx (Cond.E, f)) in
            let np = emit st Mir_type.Pred (Setcc_zx (Cond.Np, f)) in
            def (Alu (Alu.And, Sz.L, e, np))
      | Mir_op.Fcmp.Lt ->
          setcc Cond.A (emit st Mir_type.Flags (Ucomis (fsz, b, a)))
      | Mir_op.Fcmp.Le ->
          setcc Cond.Ae (emit st Mir_type.Flags (Ucomis (fsz, b, a)))
      | Mir_op.Fcmp.Unordered ->
          setcc Cond.P (emit st Mir_type.Flags (Ucomis (fsz, a, b))))
  | Mir_op.Fconvert (c, a) ->
      def
        (match c with
        | Mir_op.Fconvert.F32_to_f64 -> Cvt (Fsz.D, a)
        | Mir_op.Fconvert.F64_to_f32 -> Cvt (Fsz.S, a)
        | Mir_op.Fconvert.S64_to_f32 -> Cvtsi2s (Fsz.S, a)
        | Mir_op.Fconvert.S64_to_f64 -> Cvtsi2s (Fsz.D, a))
  | Mir_op.Ffma (a, b, c) ->
      (* SSE2 has no fused multiply-add: never a rounded multiply then add *)
      need_feature st Mir_target.Feature.Fma;
      def (Fmadd231 (fsz_of st (ty a), a, b, c))
  | Mir_op.Fto_sint a -> def (Cvtts2si (Fsz.D, a))
  | Mir_op.Funary (Mir_op.Funary.Neg, a) ->
      let fsz = fsz_of st (ty a) in
      let it, sign =
        match fsz with
        | Fsz.D -> (Mir_type.i64, Int64.min_int)
        | Fsz.S -> (Mir_type.i32, 0x8000_0000L)
      in
      let g = emit st it (Mov_imm (it, sign)) in
      let m = emit st (ty a) (Movq_from_gpr (fsz, g)) in
      def (Flogic (Flogic.Xor, fsz, a, m))
  | Mir_op.Funary (Mir_op.Funary.Sqrt, a) -> def (Sqrt (fsz_of st (ty a), a))
  | Mir_op.Funary (Mir_op.Funary.Trunc, a) ->
      need_feature st Mir_target.Feature.Sse41;
      def (Round_trunc (fsz_of st (ty a), a))
  | Mir_op.Iarith (o, a, c) -> (
      let sz = sz_of st (ty a) in
      let shift k =
        match Mir_select.const_of st.b c with
        | Some n -> def (Shift_imm (k, sz, a, Int64.to_int n))
        | None -> unsupported ()
      in
      match o with
      | Mir_op.Iarith.Add -> def (Alu (Alu.Add, sz, a, c))
      | Mir_op.Iarith.And -> def (Alu (Alu.And, sz, a, c))
      | Mir_op.Iarith.Mul -> def (Imul (sz, a, c))
      | Mir_op.Iarith.Or -> def (Alu (Alu.Or, sz, a, c))
      | Mir_op.Iarith.Shl -> shift Shift.Shl
      | Mir_op.Iarith.Shr_s -> shift Shift.Sar
      | Mir_op.Iarith.Shr_u -> shift Shift.Shr
      | Mir_op.Iarith.Sub -> def (Alu (Alu.Sub, sz, a, c))
      | Mir_op.Iarith.Xor -> def (Alu (Alu.Xor, sz, a, c)))
  | Mir_op.Icmp (c, a, b) ->
      setcc (icond st c) (emit st Mir_type.Flags (Cmp (sz_of st (ty a), a, b)))
  | Mir_op.Idiv (o, a, b) ->
      if not (Mir_type.equal (ty a) Mir_type.i64) then
        refuse st (Refusal.Width (ty a));
      let a, b = if mutated st Mutation.Division_swap then (b, a) else (a, b) in
      let q, rem =
        match o with
        | Mir_op.Idiv.Div_s -> (r (), fresh st Mir_type.i64)
        | Mir_op.Idiv.Rem_s -> (fresh st Mir_type.i64, r ())
      in
      push st [ q; rem ] (Mir_sel.Op.Machine (Cqo_idiv (a, b)))
  | Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Movsxd a)
  | Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W64, a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Movzx32 a)
  | Mir_op.Iext (k, Mir_width.W32, a)
    when Mir_type.equal (ty a) Mir_type.i8 || Mir_type.equal (ty a) Mir_type.i16
    ->
      let from = match ty a with Mir_type.Int w -> w | _ -> Mir_width.W8 in
      def (Ext { signed = k = Mir_op.Iext.Sext; from; src = a })
  | Mir_op.Iext _ -> unsupported ()
  | (Mir_op.Itrunc (Mir_width.W32, a) | Mir_op.Narrow (Mir_width.W32, a))
    when Mir_type.equal (ty a) Mir_type.i64 ->
      def (Trunc32 a)
  | Mir_op.Itrunc (((Mir_width.W8 | Mir_width.W16) as w), a)
    when Mir_type.equal (ty a) Mir_type.i32 ->
      def (Trunc_zx (w, a))
  | Mir_op.Itrunc _ | Mir_op.Narrow _ -> unsupported ()
  | Mir_op.Load { Mir_op.Access.width; addr; _ } ->
      push st ?order:i.Mir_instr.order
        [ r () ]
        (Mir_sel.Op.Machine (Load (msz_of width, address st addr)))
  | Mir_op.Pbinary (o, a, b) ->
      def
        (Alu
           ( (match o with
             | Mir_op.Pbinary.And -> Alu.And
             | Mir_op.Pbinary.Or -> Alu.Or
             | Mir_op.Pbinary.Xor -> Alu.Xor),
             Sz.L,
             a,
             b ))
  | Mir_op.Pnot a ->
      let one = emit st Mir_type.Pred (Mov_imm (Mir_type.Pred, 1L)) in
      def (Alu (Alu.Xor, Sz.L, a, one))
  | Mir_op.Ptr_add (p, d) -> def (Alu (Alu.Add, Sz.Q, p, d))
  | Mir_op.Select (p, a, c) -> (
      match ty a with
      | Mir_type.F32 | Mir_type.F64 ->
          fselect st ~result:(r ()) (fsz_of st (ty a)) p a c
      | t ->
          let f = emit st Mir_type.Flags (Test (Sz.L, p, p)) in
          def (Cmov (sz_of st t, Cond.Ne, f, a, c)))
  | Mir_op.Store ({ Mir_op.Access.width; addr; _ }, v) ->
      push st ?order:i.Mir_instr.order []
        (Mir_sel.Op.Machine (Store (msz_of width, address st addr, v)))
  | Mir_op.Undef v -> push st ?order:i.Mir_instr.order [] (Mir_sel.Op.Undef v)

let fail st (f : Mir_fail.t) =
  match
    Mir_select.store_record st.b f ~unlisted:st.unlisted ~sites:st.sites
      ~base:(fun () -> emit st Mir_type.Ptr (Lea_view Mir_select.record_view))
      ~const:(fun ty bits ->
        let bits =
          match ty with
          | Mir_type.Int w -> Mir_width.normalize w bits
          | _ -> bits
        in
        emit st ty (Mov_imm (ty, bits)))
      ~float_bits:(fun v -> emit st Mir_type.i64 (Movq_to_gpr (Fsz.D, v)))
      ~store:(fun base off v ~wide ->
        Store
          ( (if wide then Msz.Q else Msz.L),
            { Addr.base; index = None; disp = off },
            v ))
      ~one:(fun () -> emit st Mir_type.i32 (Mov_imm (Mir_type.i32, 1L)))
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
  st.b.Mir_select.origin <- Mir_origin.unknown;
  let terminator =
    match blk.Mir_block.terminator with
    | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
        let f = emit st Mir_type.Flags (Test (Sz.L, cond, cond)) in
        Mir_sel.Terminator.Branch
          {
            test = Jcc (Cond.Ne, f);
            then_ = ord_edge st then_;
            else_ = ord_edge st else_;
          }
    | Mir_terminator.Jump e -> Mir_sel.Terminator.Jump (ord_edge st e)
    | Mir_terminator.Return { Mir_return.values; order } ->
        let status = emit st Mir_type.i32 (Mov_imm (Mir_type.i32, 0L)) in
        Mir_sel.Terminator.Return
          { Mir_return.values = values @ [ status ]; order = ord st order }
    | Mir_terminator.Fail f -> fail st f
  in
  Mir_select.finish st.b terminator

let program ?mutation ?(sites = [||]) ?(unlisted = Mir_failure.Unlisted.Refused)
    ?(features = [ Mir_target.Feature.Sse2 ]) (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  let fallible = Mir_select.fallibility p in
  Err.Escape.with_escape @@ fun esc ->
  let funcs =
    List.map
      (fun (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) ->
        let st =
          {
            b = Mir_select.create f;
            esc;
            mutation;
            sites;
            unlisted;
            fallible;
            features;
          }
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
    Err.payload (S.verify { S.features; program = Mir_select.program p funcs })
  with
  | Ok v -> { selected = v; record = Mir_select.record_view }
  | Error d -> Err.Escape.throw esc (Refusal.Invalid_selection d)
