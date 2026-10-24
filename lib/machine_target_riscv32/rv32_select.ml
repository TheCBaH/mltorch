open Machine_ir
open Rv32_op
module S = Rv32_stage.Sel
open Rv32_select_st
open Rv32_select_seq
module Refusal = Rv32_select_st.Refusal
module Mutation = Rv32_select_st.Mutation

type result = { selected : S.Verified.t; record : Mir_id.View.t }

(* {1 Instructions} *)

let instr st (i : Mir_op.t Mir_instr.t) =
  st.b.Mir_select.origin <- i.Mir_instr.origin;
  let unsupported () =
    refuse st (Refusal.Operation (Mir_op.name i.Mir_instr.op))
  in
  if List.exists is_vector (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results)
  then Rv32_select_vec.vector st i
  else
    let result = match i.Mir_instr.results with [ r ] -> Some r | _ -> None in
    let r () =
      match result with
      | Some r -> r
      | None -> invalid_arg "Rv32_select: a result"
    in
    let def op = ignore (emit st ~result:(r ()) (ty (r ())) op) in
    let order = i.Mir_instr.order in
    let u = u st in
    match i.Mir_instr.op with
    | Mir_op.Addr view -> def (La view)
    | Mir_op.Bitcast (_, a) when Hashtbl.mem st.float_loads (key a) -> ()
    | Mir_op.Bitcast (t, a) -> (
        match (ty a, t) with
        | Mir_type.Int Mir_width.W32, Mir_type.F32 -> def (Fmv_w_x (u a))
        | Mir_type.F32, Mir_type.Int Mir_width.W32 -> def (Fmv_x_w (u a))
        | _ -> refuse st (Refusal.Width t))
    | Mir_op.Call (callee, args) ->
        let results = i.Mir_instr.results in
        List.iter
          (fun (v : Mir_value.t) ->
            match ty v with
            | Mir_type.Int Mir_width.W64 | Mir_type.F64 ->
                refuse st (Refusal.Width (ty v))
            | _ -> ())
          (args @ results);
        let status = fresh st Mir_type.i32 in
        push st ?order (results @ [ status ])
          (Mir_sel.Op.Machine
             (Call
                {
                  callee;
                  args = List.map u args;
                  results = List.map ty results;
                }));
        if st.fallible callee then (
          if results <> [] || st.b.Mir_select.results <> [] then
            refuse st (Refusal.Fallible_with_results callee);
          Mir_select.split_on_status st.b
            ~nonzero:(fun s -> Bnez s)
            ~status ~after:(Option.get order).Mir_order.output)
    | Mir_op.Const c when Mir_type.equal c.Mir_const.ty Mir_type.i64 ->
        const_pair st (pair st (r ())) c.Mir_const.bits
    | Mir_op.Const c -> const st ~result:(r ()) c
    | Mir_op.Copy a -> (
        match ty a with
        | Mir_type.Int Mir_width.W64 -> (* the halves are shared *) ()
        | Mir_type.F32 -> def (Fsgnj (Sgnj.J, u a, u a))
        | t when Option.is_some (Rv32.shape t) -> def (Mv (u a))
        | t -> refuse st (Refusal.Width t))
    | Mir_op.Event (e, n) -> push st ?order [] (Mir_sel.Op.Event (e, n))
    | Mir_op.Fbinary (o, a, b) -> (
        f32_only st a;
        let a = u a and b = u b in
        match o with
        | Mir_op.Fbinary.Add -> def (Fbin (Fop.Add, a, b))
        | Mir_op.Fbinary.Div -> def (Fbin (Fop.Div, a, b))
        | Mir_op.Fbinary.Max -> ignore (fmax st ~result:(r ()) a b)
        | Mir_op.Fbinary.Mul -> def (Fbin (Fop.Mul, a, b))
        | Mir_op.Fbinary.Sub -> def (Fbin (Fop.Sub, a, b)))
    | Mir_op.Fcmp (c, a, b) -> (
        f32_only st a;
        let a = u a and b = u b in
        match c with
        | Mir_op.Fcmp.Eq -> def (Fcmp (Fcmp.Eq, a, b))
        | Mir_op.Fcmp.Le -> def (Fcmp (Fcmp.Le, a, b))
        | Mir_op.Fcmp.Lt -> def (Fcmp (Fcmp.Lt, a, b))
        | Mir_op.Fcmp.Unordered ->
            let both =
              alu st Alu.And
                (emit st Mir_type.Pred (Fcmp (Fcmp.Eq, a, a)))
                (emit st Mir_type.Pred (Fcmp (Fcmp.Eq, b, b)))
            in
            ignore (pnot st ~result:(r ()) both))
    | Mir_op.Fconvert (Mir_op.Fconvert.S32_to_f32, a) -> def (Fcvt_s_w (u a))
    | Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f32, a) ->
        let p = pair st a in
        if p.sext then def (Fcvt_s_w p.lo)
        else ignore (i64_to_f32 st ~result:(r ()) p)
    | Mir_op.Fconvert _ | Mir_op.Fto_sint _ ->
        refuse st (Refusal.Width Mir_type.F64)
    | Mir_op.Ffma (a, b, c) ->
        f32_only st a;
        def (Fmadd (u a, u b, u c))
    | Mir_op.Funary (o, a) -> (
        f32_only st a;
        let a = u a in
        match o with
        | Mir_op.Funary.Neg -> def (Fsgnj (Sgnj.Jn, a, a))
        | Mir_op.Funary.Sqrt -> def (Fsqrt a)
        | Mir_op.Funary.Trunc -> ignore (ftrunc st ~result:(r ()) a))
    | Mir_op.Iarith (o, a, b) -> (
        match ty a with
        | Mir_type.Int Mir_width.W32 ->
            ignore (iarith32 st ~result:(r ()) o (u a) (u b))
        | Mir_type.Int Mir_width.W64 ->
            iarith64 st o (pair st a) b (pair st (r ()))
        | Mir_type.Int ((Mir_width.W8 | Mir_width.W16) as w) ->
            let signed = o = Mir_op.Iarith.Shr_s in
            let x = widen st ~signed (u a) in
            let y =
              if Mir_op.Iarith.is_shift o then u b
              else widen st ~signed:false (u b)
            in
            let y = if Mir_op.Iarith.is_shift o then y else y in
            let w32 =
              if Mir_op.Iarith.is_shift o then
                match const_of st b with
                | Some c ->
                    alui st
                      (match o with
                      | Mir_op.Iarith.Shl -> Alui.Slli
                      | Mir_op.Iarith.Shr_s -> Alui.Srai
                      | _ -> Alui.Srli)
                      x (Int64.logand c 31L)
                | None ->
                    refuse st (Refusal.Operation "shift by a variable amount")
              else iarith32 st o x y
            in
            def (Trunc (w, w32))
        | t -> refuse st (Refusal.Width t))
    | Mir_op.Icmp (c, a, b) -> (
        match ty a with
        | Mir_type.Int Mir_width.W64 ->
            ignore (compare64 st ~result:(r ()) c (pair st a) (pair st b))
        | Mir_type.Int Mir_width.W32 ->
            ignore (compare32 st ~result:(r ()) c (u a) (u b))
        | Mir_type.Int (Mir_width.W8 | Mir_width.W16) | Mir_type.Pred ->
            let signed =
              match c with
              | Mir_op.Icmp.Sle | Mir_op.Icmp.Slt -> true
              | _ -> false
            in
            ignore
              (compare32 st ~result:(r ()) c
                 (widen st ~signed (u a))
                 (widen st ~signed (u b)))
        | t -> refuse st (Refusal.Width t))
    | Mir_op.Idiv (o, a, b) -> (
        let o =
          match o with
          | Mir_op.Idiv.Div_s -> Alu.Div
          | Mir_op.Idiv.Rem_s -> Alu.Rem
        in
        match ty a with
        | Mir_type.Int Mir_width.W32 -> def (Alu (o, u a, u b))
        | Mir_type.Int Mir_width.W64 -> (
            (* two sign-extended words divide as words, unless the quotient of
             the most negative by -1 is wanted: a constant divisor other than
             -1 rules that out *)
            let pa = pair st a
            and pb = pair st b in
            let r = pair st (r ()) in
            match const_of st b with
            | Some d when pa.sext && pb.sext && not (Int64.equal d (-1L)) ->
                let q =
                  emit st ~result:r.lo Mir_type.i32 (Alu (o, pa.lo, pb.lo))
                in
                ignore
                  (emit st ~result:r.hi Mir_type.i32 (Alui (Alui.Srai, q, 31L)))
            | _ -> refuse st (Refusal.Operation "an i64 division"))
        | t -> refuse st (Refusal.Width t))
    | Mir_op.Iext (k, w, a) -> (
        let signed = k = Mir_op.Iext.Sext in
        match (w, ty a) with
        | Mir_width.W64, Mir_type.Int Mir_width.W32 ->
            (* the low half is the operand itself *)
            let p = pair st (r ()) in
            if signed then
              ignore
                (emit st ~result:p.hi Mir_type.i32 (Alui (Alui.Srai, u a, 31L)))
            else ignore (materialize st ~result:p.hi Mir_type.i32 0L)
        | Mir_width.W64, _ ->
            let x = widen st ~signed (u a) in
            let p = pair st (r ()) in
            ignore (emit st ~result:p.lo Mir_type.i32 (Mv x));
            if signed then
              ignore
                (emit st ~result:p.hi Mir_type.i32 (Alui (Alui.Srai, x, 31L)))
            else ignore (materialize st ~result:p.hi Mir_type.i32 0L)
        | Mir_width.W32, Mir_type.Int ((Mir_width.W8 | Mir_width.W16) as from)
          ->
            def (if signed then Sext (from, u a) else Zext (u a))
        | Mir_width.W16, Mir_type.Int Mir_width.W8 ->
            def (Trunc (Mir_width.W16, widen st ~signed (u a)))
        | _ -> unsupported ())
    | (Mir_op.Itrunc (Mir_width.W32, a) | Mir_op.Narrow (Mir_width.W32, a))
      when is_i64 a ->
        (* its uses read the low half *)
        ()
    | Mir_op.Itrunc (((Mir_width.W8 | Mir_width.W16) as w), a)
    | Mir_op.Narrow (((Mir_width.W8 | Mir_width.W16) as w), a) ->
        let x = if is_i64 a then lo st a else widen st ~signed:false (u a) in
        def (Trunc (w, x))
    | Mir_op.Itrunc _ | Mir_op.Narrow _ -> unsupported ()
    | Mir_op.Load { Mir_op.Access.width = Mir_width.W64; addr; _ } ->
        let p = pair st (r ()) in
        let base, k = displaced st ~extra:4L addr in
        (* two loads in a row, chained on the order the access threads *)
        let o = Option.get order in
        let mid = fresh st Mir_type.Order in
        push st
          ~order:{ o with Mir_order.output = mid }
          [ p.lo ]
          (Mir_sel.Op.Machine (Load (Msz.W, base, k)));
        push st
          ~order:{ Mir_order.input = mid; output = o.Mir_order.output }
          [ p.hi ]
          (Mir_sel.Op.Machine (Load (Msz.W, base, Int64.add k 4L)))
    | Mir_op.Load { Mir_op.Access.width; addr; _ } -> (
        let base, k = displaced st addr in
        match Hashtbl.find_opt st.float_loads (key (r ())) with
        | Some f -> push st ?order [ f ] (Mir_sel.Op.Machine (Flw (base, k)))
        | None ->
            let m =
              match width with
              | Mir_width.W8 -> Msz.B
              | Mir_width.W16 -> Msz.H
              | _ -> Msz.W
            in
            push st ?order [ r () ] (Mir_sel.Op.Machine (Load (m, base, k))))
    | Mir_op.Pbinary (o, a, b) ->
        def
          (Alu
             ( (match o with
               | Mir_op.Pbinary.And -> Alu.And
               | Mir_op.Pbinary.Or -> Alu.Or
               | Mir_op.Pbinary.Xor -> Alu.Xor),
               u a,
               u b ))
    | Mir_op.Pnot a -> ignore (pnot st ~result:(r ()) (u a))
    | Mir_op.Ptr_add (p, d) -> (
        let d = if is_i64 d then lo st d else u d in
        match const_of st d with
        | Some k when imm12 (s32 k) -> def (Addi (u p, s32 k))
        | _ -> def (Alu (Alu.Add, u p, d)))
    | Mir_op.Select (p, a, b) -> (
        let p = u p in
        match ty a with
        | Mir_type.Int Mir_width.W64 ->
            let pa = pair st a and pb = pair st b and pr = pair st (r ()) in
            ignore (emit st ~result:pr.lo Mir_type.i32 (Sel (p, pa.lo, pb.lo)));
            ignore (emit st ~result:pr.hi Mir_type.i32 (Sel (p, pa.hi, pb.hi)))
        | Mir_type.F32 -> def (Fsel (p, u a, u b))
        | t when Option.is_some (Rv32.shape t) -> def (Sel (p, u a, u b))
        | t -> refuse st (Refusal.Width t))
    | Mir_op.Store ({ Mir_op.Access.width = Mir_width.W64; addr; _ }, v) ->
        let p = pair st v in
        let base, k = displaced st ~extra:4L addr in
        push st ?order [] (Mir_sel.Op.Machine (Sw_pair (base, k, p.lo, p.hi)))
    | Mir_op.Store ({ Mir_op.Access.width; addr; _ }, v) -> (
        let base, k = displaced st addr in
        match Hashtbl.find_opt st.b.Mir_select.defs (key v) with
        | Some (Mir_op.Bitcast (_, f))
          when width = Mir_width.W32 && Mir_type.equal (ty f) Mir_type.F32 ->
            push st ?order [] (Mir_sel.Op.Machine (Fsw (base, k, u f)))
        | _ ->
            let m =
              match width with
              | Mir_width.W8 -> Msz.B
              | Mir_width.W16 -> Msz.H
              | _ -> Msz.W
            in
            push st ?order [] (Mir_sel.Op.Machine (Store (m, base, k, u v))))
    | Mir_op.Undef v -> push st ?order [] (Mir_sel.Op.Undef v)
    | Mir_op.Vconcat _ | Mir_op.Vextract _ | Mir_op.Vinsert _ | Mir_op.Vload _
    | Mir_op.Vslice _ | Mir_op.Vsplat _ | Mir_op.Vstore _ ->
        unsupported ()

(* {1 Blocks} *)

(* A value as the words that carry it: an i64 its two halves. *)
let words st (v : Mir_value.t) =
  if is_i64 v then
    let p = pair st v in
    [ p.lo; p.hi ]
  else
    match Rv32.shape (ty v) with
    | Some _ -> [ u st v ]
    | None -> refuse st (Refusal.Width (ty v))

let edge st (e : Mir_edge.t) =
  let e = Mir_select.ord_edge st.b e in
  { e with Mir_edge.args = List.concat_map (words st) e.Mir_edge.args }

(* The failure record stores, then status 1. *)
let fail st (f : Mir_fail.t) =
  st.b.Mir_select.origin <- f.Mir_fail.origin;
  let base () = emit st Mir_type.Ptr (La Mir_select.record_view) in
  let wide_const = Hashtbl.create 8 in
  match
    Mir_select.store_record st.b f ~unlisted:st.unlisted ~sites:st.sites ~base
      ~const:(fun ty bits ->
        match ty with
        | Mir_type.Int Mir_width.W64 ->
            (* a constant word: its halves, under a name only the store reads *)
            let v = fresh st Mir_type.i64 in
            let p =
              {
                lo = fresh st Mir_type.i32;
                hi = fresh st Mir_type.i32;
                sext = false;
              }
            in
            const_pair st p bits;
            Hashtbl.replace wide_const (key v) p;
            v
        | _ -> materialize st ty bits)
      ~float_bits:(fun v -> refuse st (Refusal.Width (ty v)))
      ~store:(fun base off v ~wide ->
        if wide then
          let p =
            match Hashtbl.find_opt wide_const (key v) with
            | Some p -> p
            | None -> pair st v
          in
          Sw_pair (base, off, p.lo, p.hi)
        else Store (Msz.W, base, off, u st v))
      ~one:(fun () -> materialize st Mir_type.i32 1L)
      ?skip:
        (if mutated st Mutation.Missing_failure_word then Some `Last else None)
      ()
  with
  | Some term -> term
  | None -> refuse st (Refusal.Missing_site f.Mir_fail.failure)

let block st (blk : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
  Mir_select.start st.b blk.Mir_block.id
    (List.concat_map (words st) blk.Mir_block.params)
    blk.Mir_block.order;
  let body =
    if mutated st Mutation.Contract then Mir_select.contract blk.Mir_block.body
    else blk.Mir_block.body
  in
  List.iter (instr st) body;
  st.b.Mir_select.origin <-
    (match List.rev body with
    | (last : Mir_op.t Mir_instr.t) :: _ -> last.Mir_instr.origin
    | [] -> Mir_origin.unknown);
  let terminator =
    match blk.Mir_block.terminator with
    | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
        (* a compare of two words in this block is the branch's own *)
        let test =
          match
            List.find_opt
              (fun (i : Mir_op.t Mir_instr.t) ->
                List.exists (Mir_value.equal cond) i.Mir_instr.results)
              body
          with
          | Some { Mir_instr.op = Mir_op.Icmp (c, a, b); _ }
            when Mir_type.equal (ty a) Mir_type.i32 -> (
              let a = u st a and b = u st b in
              match c with
              | Mir_op.Icmp.Eq -> Bcmp (Bcond.Eq, a, b)
              | Mir_op.Icmp.Ne -> Bcmp (Bcond.Ne, a, b)
              | Mir_op.Icmp.Slt -> Bcmp (Bcond.Lt, a, b)
              | Mir_op.Icmp.Sle -> Bcmp (Bcond.Ge, b, a)
              | Mir_op.Icmp.Ult -> Bcmp (Bcond.Ltu, a, b)
              | Mir_op.Icmp.Ule -> Bcmp (Bcond.Geu, b, a))
          | _ -> Bnez (u st cond)
        in
        Mir_sel.Terminator.Branch
          { test; then_ = edge st then_; else_ = edge st else_ }
    | Mir_terminator.Jump e -> Mir_sel.Terminator.Jump (edge st e)
    | Mir_terminator.Return { Mir_return.values; order } ->
        let status = materialize st Mir_type.i32 0L in
        Mir_sel.Terminator.Return
          {
            Mir_return.values = List.concat_map (words st) values @ [ status ];
            order = ord st order;
          }
    | Mir_terminator.Fail f -> fail st f
  in
  Mir_select.finish st.b terminator

(* The features the selected forms use: Zve32f when any is a vector form. *)
let features funcs =
  let vector =
    List.exists
      (fun (f : (_, _) Mir_func.t) ->
        List.exists
          (fun (b : (_, _) Mir_block.t) ->
            List.exists
              (fun (i : _ Mir_instr.t) ->
                match i.Mir_instr.op with
                | Mir_sel.Op.Machine (Rv32_op.Vec _) -> true
                | _ -> false)
              b.Mir_block.body)
          f.Mir_func.blocks)
      funcs
  in
  if vector then [ Mir_target.Feature.Zve32f ] else []

let program ?mutation ?(register_bytes = 16L) ?(sites = [||])
    ?(unlisted = Mir_failure.Unlisted.Refused) (g : Mir_verify.Generic.t) =
  Err.Escape.with_escape @@ fun esc ->
  let g =
    match Err.payload (Mir_vsplit.program ~register_bytes g) with
    | Ok g -> g
    | Error r -> Err.Escape.throw esc (Refusal.Vector r)
  in
  let p =
    Mir_verify.Generic.program
      (Mir_cse.program (Mir_narrow.program (Mir_offsets.program g)))
  in
  let fallible = Mir_select.fallibility p in
  let funcs =
    List.map
      (fun (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) ->
        List.iter
          (fun t ->
            match t with
            | Mir_type.Int Mir_width.W64 | Mir_type.F64 ->
                Err.Escape.throw esc (Refusal.Width t)
            | _ -> ())
          f.Mir_func.results;
        let st =
          {
            b = Mir_select.create f;
            esc;
            mutation;
            sites;
            unlisted;
            fallible;
            pairs = Hashtbl.create 64;
            subst = Hashtbl.create 64;
            float_loads = float_loads f;
            int_slices = Mir_vsplit.widening_slices f;
            shift_counts = Mir_vsplit.shift_counts f;
          }
        in
        name_pairs st f;
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
  let program =
    {
      (Mir_select.program p funcs) with
      Mir_program.data_model = Mir_layout.Data_model.Ilp32_le;
    }
  in
  match
    Err.payload
      (S.verify
         (S.prune
            ~ignore_terminators:(mutation = Some Mutation.Pruned_live)
            { S.features = features funcs; program }))
  with
  | Ok v -> { selected = v; record = Mir_select.record_view }
  | Error d -> Err.Escape.throw esc (Refusal.Invalid_selection d)
