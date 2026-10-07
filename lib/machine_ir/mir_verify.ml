(* The generic verifier: shared structure, then the generic stage's own
   semantic rules — constant shift counts, guard evidence for the
   domain-restricted operations, static access permissions, failure payload
   schemas and return signatures. *)

module P = Mir_diagnostic.Problem

module Stage = struct
  type op = Mir_op.t
  type term = Mir_terminator.t

  let stage = Mir_diagnostic.Stage.Generic
  let operands = Mir_op.operands
  let ordered op = Mir_op.Effect.ordered (Mir_op.effect_class op)

  let typing (cx : Mir_check.context) op =
    Result.map_error
      (fun e -> P.Typing e)
      (Mir_typing.check ~signature:cx.Mir_check.signature
         ~view:(fun v -> Option.is_some (cx.Mir_check.view v))
         op)

  let edges = Mir_terminator.edges

  let term_values = function
    | Mir_terminator.Branch { Mir_branch.cond; _ } -> [ cond ]
    | Mir_terminator.Fail { Mir_fail.payload; _ } -> payload
    | Mir_terminator.Jump _ -> []
    | Mir_terminator.Return { Mir_return.values; _ } -> values

  let term_order = function
    | Mir_terminator.Fail { Mir_fail.order; _ }
    | Mir_terminator.Return { Mir_return.order; _ } ->
        Some order
    | Mir_terminator.Branch _ | Mir_terminator.Jump _ -> None

  let types vs = List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) vs
  let same_types a b = List.equal Mir_type.equal a b

  let check_term _cx ~results = function
    | Mir_terminator.Branch { Mir_branch.cond; _ } ->
        if Mir_type.equal cond.Mir_value.ty Mir_type.Pred then Ok ()
        else Error (P.Branch_condition cond.Mir_value.id)
    | Mir_terminator.Fail { Mir_fail.failure; payload; _ } ->
        if same_types (Mir_failure.payload failure) (types payload) then Ok ()
        else Error (P.Payload_mismatch failure)
    | Mir_terminator.Jump _ -> Ok ()
    | Mir_terminator.Return { Mir_return.values; _ } ->
        if same_types results (types values) then Ok ()
        else Error P.Return_mismatch
end

module C = Mir_check.Make (Stage)

(* The operation defining a value, when an instruction defines it. *)
let def_op (a : Mir_op.t Mir_check.Analysis.t) (v : Mir_value.t) =
  match Mir_id.Value.Map.find_opt v.Mir_value.id a.Mir_check.Analysis.defs with
  | Some (Mir_check.Def.Result (i, _), _, _) -> Some i.Mir_instr.op
  | _ -> None

let const_bits a v =
  match def_op a v with
  | Some (Mir_op.Const c) -> Some c.Mir_const.bits
  | _ -> None

(* The predicate facts that hold on entry to [block]: for every block on its
   dominator chain entered from a single branching predecessor through exactly
   one of that branch's edges, the condition with that edge's polarity, closed
   under [and]/[or]/[not]. *)
let facts (a : Mir_op.t Mir_check.Analysis.t)
    (blocks : (Mir_op.t, Mir_terminator.t) Mir_block.t Mir_id.Block.Map.t) block
    =
  let g = a.Mir_check.Analysis.graph in
  let edge_fact b =
    match Mir_graph.preds g b with
    | [ p ] -> (
        match Mir_id.Block.Map.find_opt p blocks with
        | Some
            {
              Mir_block.terminator =
                Mir_terminator.Branch { Mir_branch.cond; then_; else_ };
              _;
            } -> (
            let t = Mir_id.Block.equal then_.Mir_edge.target b
            and e = Mir_id.Block.equal else_.Mir_edge.target b in
            match (t, e) with
            | true, false -> Some (cond, true)
            | false, true -> Some (cond, false)
            | _ -> None)
        | _ -> None)
    | _ -> None
  in
  let rec chain b acc =
    let acc = match edge_fact b with Some f -> f :: acc | None -> acc in
    match Mir_graph.idom g b with Some d -> chain d acc | None -> acc
  in
  let rec close acc = function
    | [] -> acc
    | ((v : Mir_value.t), pol) :: rest -> (
        let acc = (v.Mir_value.id, pol) :: acc in
        match (def_op a v, pol) with
        | Some (Mir_op.Pbinary (Mir_op.Pbinary.And, x, y)), true
        | Some (Mir_op.Pbinary (Mir_op.Pbinary.Or, x, y)), false ->
            close acc ((x, pol) :: (y, pol) :: rest)
        | Some (Mir_op.Pnot x), _ -> close acc ((x, not pol) :: rest)
        | _ -> close acc rest)
  in
  close [] (chain block [])

(* A fact proves [x <> k] (an integer) when some predicate known true is
   [x <> k], or known false is [x == k]; or [x] is a different constant. *)
let proves_ne a facts (x : Mir_value.t) k =
  (match const_bits a x with Some b -> not (Int64.equal b k) | None -> false)
  || List.exists
       (fun (id, pol) ->
         let v = { Mir_value.id; ty = Mir_type.Pred } in
         match def_op a v with
         | Some (Mir_op.Icmp (c, l, r)) ->
             let matches =
               (Mir_value.equal l x && const_bits a r = Some k)
               || (Mir_value.equal r x && const_bits a l = Some k)
             in
             matches
             && ((c = Mir_op.Icmp.Ne && pol) || (c = Mir_op.Icmp.Eq && not pol))
         | _ -> false)
       facts

(* [not (x == kx && y == ky)]: either inequality, or the conjunction known
   false. *)
let proves_not_both a facts x kx y ky =
  proves_ne a facts x kx || proves_ne a facts y ky
  || List.exists
       (fun (id, pol) ->
         let v = { Mir_value.id; ty = Mir_type.Pred } in
         (not pol)
         &&
         match def_op a v with
         | Some (Mir_op.Pbinary (Mir_op.Pbinary.And, p, q)) ->
             let is_eq (p : Mir_value.t) z k =
               match def_op a p with
               | Some (Mir_op.Icmp (Mir_op.Icmp.Eq, l, r)) ->
                   (Mir_value.equal l z && const_bits a r = Some k)
                   || (Mir_value.equal r z && const_bits a l = Some k)
               | _ -> false
             in
             (is_eq p x kx && is_eq q y ky) || (is_eq q x kx && is_eq p y ky)
         | _ -> false)
       facts

let two63 = Int64.bits_of_float 9223372036854775808.
let neg_two63 = Int64.bits_of_float (-9223372036854775808.)

(* [lo <= x] and [x < hi] known true, which a NaN fails. *)
let proves_f64_range a facts (x : Mir_value.t) =
  let known pred_shape =
    List.exists
      (fun (id, pol) ->
        pol && pred_shape (def_op a { Mir_value.id; ty = Mir_type.Pred }))
      facts
  in
  known (function
    | Some (Mir_op.Fcmp (Mir_op.Fcmp.Le, l, r)) ->
        Mir_value.equal r x && const_bits a l = Some neg_two63
    | _ -> false)
  && known (function
    | Some (Mir_op.Fcmp (Mir_op.Fcmp.Lt, l, r)) ->
        Mir_value.equal l x && const_bits a r = Some two63
    | _ -> false)

(* The view an address is statically derived from, through pointer offsets. *)
let rec root_view a (v : Mir_value.t) =
  match def_op a v with
  | Some (Mir_op.Addr view) -> Some view
  | Some (Mir_op.Ptr_add (base, _)) | Some (Mir_op.Copy base) ->
      root_view a base
  | _ -> None

let semantic esc (cx : Mir_check.context)
    (f : (Mir_op.t, Mir_terminator.t) Mir_func.t)
    (a : Mir_op.t Mir_check.Analysis.t) =
  let blocks =
    List.fold_left
      (fun m (b : (_, _) Mir_block.t) ->
        Mir_id.Block.Map.add b.Mir_block.id b m)
      Mir_id.Block.Map.empty f.Mir_func.blocks
  in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      let facts = lazy (facts a blocks b.Mir_block.id) in
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          let reject p =
            Mir_check.make_reject esc Mir_diagnostic.Stage.Generic
              ~func:f.Mir_func.id ~block:b.Mir_block.id ~instr:i.Mir_instr.id p
          in
          let permit addr ok =
            match root_view a addr with
            | Some view -> (
                match cx.Mir_check.view view with
                | Some v when ok v.Mir_view.perm -> ()
                | _ -> reject (P.Permission view))
            | None -> ()
          in
          match i.Mir_instr.op with
          | Mir_op.Iarith (op, x, count) when Mir_op.Iarith.is_shift op -> (
              match (x.Mir_value.ty, const_bits a count) with
              | Mir_type.Int w, Some c
                when Int64.compare c 0L >= 0
                     && Int64.compare c (Int64.of_int (Mir_width.bits w)) < 0 ->
                  ()
              | _ -> reject P.Shift_count)
          | Mir_op.Idiv (_, x, y) -> (
              match x.Mir_value.ty with
              | Mir_type.Int w ->
                  let f = Lazy.force facts in
                  let min = Mir_width.normalize w (Mir_width.min_signed w)
                  and minus_one = Mir_width.normalize w (-1L) in
                  if
                    not
                      (proves_ne a f y 0L
                      && proves_not_both a f x min y minus_one)
                  then reject P.Unproven_domain
              | _ -> reject P.Unproven_domain)
          | Mir_op.Fto_sint x ->
              if not (proves_f64_range a (Lazy.force facts) x) then
                reject P.Unproven_domain
          | Mir_op.Load { Mir_op.Access.addr; _ } ->
              permit addr Mir_view.readable
          | Mir_op.Store ({ Mir_op.Access.addr; _ }, _) ->
              permit addr Mir_view.writable
          | Mir_op.Undef view -> (
              match cx.Mir_check.view view with
              | Some v when Mir_view.writable v.Mir_view.perm -> ()
              | _ -> reject (P.Permission view))
          | Mir_op.Addr _ | Mir_op.Bitcast _ | Mir_op.Call _ | Mir_op.Const _
          | Mir_op.Copy _ | Mir_op.Event _ | Mir_op.Fbinary _ | Mir_op.Fcmp _
          | Mir_op.Fconvert _ | Mir_op.Ffma _ | Mir_op.Funary _
          | Mir_op.Iarith _ | Mir_op.Icmp _ | Mir_op.Iext _ | Mir_op.Itrunc _
          | Mir_op.Narrow _ | Mir_op.Pbinary _ | Mir_op.Pnot _
          | Mir_op.Ptr_add _ | Mir_op.Select _ ->
              ())
        b.Mir_block.body)
    f.Mir_func.blocks

module Generic = struct
  type t = Mir_program.generic

  let program t = t
  let revision (t : t) = t.Mir_program.revision
end

let generic (p : Mir_program.generic) =
  Err.Escape.with_escape @@ fun esc ->
  let cx, analyses = Err.Escape.or_throw esc (C.program p) in
  List.iter (fun (f, a) -> semantic esc cx f a) analyses;
  (p : Generic.t)
