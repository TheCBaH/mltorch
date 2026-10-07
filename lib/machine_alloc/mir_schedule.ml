(* Block-local instruction scheduling of a selected program, before liveness
   and allocation. A block's body is reordered within the dependences of its
   source order: a value's definition before its uses, the order chain (every
   memory access, call, event and [undef] keeps its place relative to the
   others), and condition state — a writer of the flags stays on the side of a
   [Flags] value's live range it started on. Nothing crosses a block, so no
   load or check is speculated. A schedule is checked against the dependences
   of the order it replaced and the selected verifier runs again; a changed
   order is a new revision, whose liveness and allocation are recomputed. *)

open Machine_ir

module Policy = struct
  type t =
    | Reverse
        (** the latest ready instruction first: the legal order farthest from
            the source's, for the evidence suite *)
    | Sink
        (** a pure instruction whose results stay in its block moved down to
            just before its first dependent *)
    | Source  (** the selected order, unchanged *)

  let name = function
    | Reverse -> "reverse"
    | Sink -> "sink"
    | Source -> "source"
end

(** Fault injection for the evidence suite: each drops one class of dependence
    from the graph the scheduler orders by. No consumer passes one. *)
module Mutation = struct
  type t =
    | Data  (** a use not ordered after its definition *)
    | Flags  (** a flags writer free to enter a [Flags] value's range *)
    | Order  (** the order chain *)
end

(* Where a scheduled block breaks the order it replaced. *)
module Violation = struct
  type t =
    | Not_permutation of Mir_id.Block.t
    | Reversed of {
        block : Mir_id.Block.t;
        first : Mir_id.Instr.t;  (** what must come first *)
        second : Mir_id.Instr.t;
      }

  let pp fmt = function
    | Not_permutation b ->
        Fmt.pf fmt "%a is not a permutation of its body" Mir_id.Block.pp b
    | Reversed { block; first; second } ->
        Fmt.pf fmt "%a: %a scheduled after %a, which depends on it"
          Mir_id.Block.pp block Mir_id.Instr.pp first Mir_id.Instr.pp second
end

module Refusal = struct
  type t = Dependence of Violation.t | Invalid of Mir_diagnostic.t

  let pp fmt = function
    | Dependence v -> Fmt.pf fmt "dependence: %a" Violation.pp v
    | Invalid d -> Fmt.pf fmt "selected verifier: %a" Mir_diagnostic.pp d
end

module Make (T : Mir_sel.TARGET) = struct
  module S = Mir_sel.Make (T)
  module L = Mir_liveness.Make (T)

  type instr = S.Stage.op Mir_instr.t
  type block = (S.Stage.op, S.Stage.term) Mir_block.t

  let key (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id
  let is_flags (v : Mir_value.t) = Mir_type.equal v.Mir_value.ty Mir_type.Flags
  let uses (i : instr) = S.Stage.operands i.Mir_instr.op

  let writes_flags (i : instr) =
    match i.Mir_instr.op with
    | Mir_sel.Op.Machine o -> T.writes_flags o
    | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> false

  (* The dependence edges of a body, [(a, b)] with [a < b] by index: [b] must
     follow [a]. *)
  let dependencies ?mutation (body : instr array) =
    let def = Hashtbl.create 16 in
    Array.iteri
      (fun k (i : instr) ->
        List.iter (fun v -> Hashtbl.replace def (key v) k) i.Mir_instr.results;
        Option.iter
          (fun (o : Mir_order.t) ->
            Hashtbl.replace def (key o.Mir_order.output) k)
          i.Mir_instr.order)
      body;
    let edges = ref [] in
    let add a b = if a <> b then edges := (a, b) :: !edges in
    let from v b =
      Option.iter (fun a -> add a b) (Hashtbl.find_opt def (key v))
    in
    Array.iteri
      (fun b (i : instr) ->
        if mutation <> Some Mutation.Data then
          List.iter (fun v -> from v b) (uses i);
        if mutation <> Some Mutation.Order then
          Option.iter
            (fun (o : Mir_order.t) -> from o.Mir_order.input b)
            i.Mir_instr.order)
      body;
    (if mutation <> Some Mutation.Flags then
       let users v =
         List.filter_map Fun.id
           (List.mapi
              (fun k (i : instr) ->
                if List.exists (Mir_value.equal v) (uses i) then Some k
                else None)
              (Array.to_list body))
       in
       Array.iteri
         (fun d (i : instr) ->
           List.iter
             (fun r ->
               if is_flags r then
                 let us = users r in
                 let last = List.fold_left max d us in
                 Array.iteri
                   (fun w wi ->
                     if w <> d && writes_flags wi then
                       if w < d then add w d
                       else if w > last then List.iter (fun u -> add u w) us)
                   body)
             i.Mir_instr.results)
         body);
    List.sort_uniq compare !edges

  (* List scheduling: a ready instruction — every dependence placed — of least
     [priority] next. [Sink] anchors each instruction that is pure, leaves the
     condition state alone and whose results the block alone reads at its first
     dependent's source place, the rest at their own, ties in source order: a
     pure value is computed just before the place it is first needed. Anchoring
     at the dependent's own anchor instead — sinking a whole pure chain to its
     consumer — spilled more on the model kernels. *)
  let list_schedule ?mutation policy ~escapes (b : block) =
    let body = Array.of_list b.Mir_block.body in
    let n = Array.length body in
    let preds = Array.make n 0 and succs = Array.make n [] in
    List.iter
      (fun (a, c) ->
        preds.(c) <- preds.(c) + 1;
        succs.(a) <- c :: succs.(a))
      (dependencies ?mutation body);
    let sinkable (i : instr) =
      (not (S.Stage.ordered i.Mir_instr.op))
      && (not (writes_flags i))
      && List.for_all
           (fun (v : Mir_value.t) ->
             not (Mir_id.Value.Set.mem v.Mir_value.id escapes))
           i.Mir_instr.results
    in
    let priority =
      Array.init n (fun k ->
          match policy with
          | Policy.Reverse -> (-k, 0)
          | Policy.Sink -> (
              match succs.(k) with
              | c :: cs when sinkable body.(k) -> (List.fold_left min c cs, k)
              | _ -> (k, k))
          | Policy.Source -> (k, 0))
    in
    let ready =
      ref (List.filter (fun k -> preds.(k) = 0) (List.init n Fun.id))
    in
    let out = ref [] in
    while !ready <> [] do
      let k =
        List.fold_left
          (fun j k -> if compare priority.(k) priority.(j) < 0 then k else j)
          (List.hd !ready) !ready
      in
      ready := List.filter (( <> ) k) !ready;
      out := body.(k) :: !out;
      List.iter
        (fun c ->
          preds.(c) <- preds.(c) - 1;
          if preds.(c) = 0 then ready := c :: !ready)
        succs.(k)
    done;
    List.rev !out

  (* The selected program reordered by [policy], unchecked. *)
  let reorder ?mutation policy (sel : S.Verified.t) =
    let s = S.Verified.selected sel in
    let p = s.S.program in
    let changed = ref false in
    let funcs =
      List.map
        (fun (f : (S.Stage.op, S.Stage.term) Mir_func.t) ->
          let _, live_out = L.sets f in
          let blocks =
            List.map
              (fun (b : block) ->
                match policy with
                | Policy.Source -> b
                | Policy.Reverse | Policy.Sink ->
                    let escapes =
                      Mir_id.Value.Set.union (live_out b.Mir_block.id)
                        (L.ids (L.term_uses b))
                    in
                    let body = list_schedule ?mutation policy ~escapes b in
                    if not (List.equal ( == ) body b.Mir_block.body) then
                      changed := true;
                    { b with Mir_block.body })
              f.Mir_func.blocks
          in
          { f with Mir_func.blocks })
        p.Mir_program.funcs
    in
    if not !changed then s
    else
      {
        s with
        S.program =
          {
            p with
            Mir_program.funcs;
            revision =
              Mir_id.Revision.of_int
                (Mir_id.Revision.to_int p.Mir_program.revision + 1);
          };
      }

  (* Whether [after] permutes each body of [before] within its dependences. *)
  let check ~(before : S.Verified.t) ~(after : S.selected) =
    let ( let* ) = Result.bind in
    let instr_key (i : instr) = Mir_id.Instr.to_int i.Mir_instr.id in
    List.fold_left
      (fun acc ((f : (S.Stage.op, S.Stage.term) Mir_func.t), g) ->
        List.fold_left
          (fun acc ((b : block), (c : block)) ->
            let* () = acc in
            let block = b.Mir_block.id in
            let ids l = List.sort compare (List.map instr_key l) in
            if ids b.Mir_block.body <> ids c.Mir_block.body then
              Error (Violation.Not_permutation block)
            else
              let at = Hashtbl.create 16 in
              List.iteri
                (fun k i -> Hashtbl.replace at (instr_key i) k)
                c.Mir_block.body;
              let body = Array.of_list b.Mir_block.body in
              List.fold_left
                (fun acc (x, y) ->
                  let* () = acc in
                  if
                    Hashtbl.find at (instr_key body.(x))
                    < Hashtbl.find at (instr_key body.(y))
                  then Ok ()
                  else
                    Error
                      (Violation.Reversed
                         {
                           block;
                           first = body.(x).Mir_instr.id;
                           second = body.(y).Mir_instr.id;
                         }))
                (Ok ()) (dependencies body))
          acc
          (List.combine f.Mir_func.blocks g.Mir_func.blocks))
      (Ok ())
      (List.combine (S.Verified.selected before).S.program.Mir_program.funcs
         after.S.program.Mir_program.funcs)

  let schedule ?mutation policy sel =
    let after = reorder ?mutation policy sel in
    if after == S.Verified.selected sel then Ok sel
    else
      match check ~before:sel ~after with
      | Error v -> Error (Refusal.Dependence v)
      | Ok () ->
          Result.map_error
            (fun d -> Refusal.Invalid d)
            (Err.payload (S.Verified.verify after))
end
