(* See recipe.mli. *)

open Graph_ir

(* Takes the [Side.S] rather than the dialect, so the snapshot [existing]
   resolves against is the SAME module [Rewrite] holds — not a second
   application that would then have to be proven equal to it. *)
module Make (S : Side.S) = struct
  module Snap = S.Snapshot

  (* Both erase to a raw id; the distinction is the type, exactly as for
     [Correspondence.id]. A source edge additionally witnesses that [existing]
     found it in the graph being edited. *)
  type 'v fresh = Tensor_id.t
  type 'v source = Tensor_id.t
  type 'v target = Fresh of 'v fresh | Preserved of 'v source
  type 'v edit_edge = New of 'v fresh | Old of 'v source

  let raw_fresh id = id
  let raw_source id = id
  let raw_target = function Fresh id -> id | Preserved id -> id
  let raw_edit_edge = function New id -> id | Old id -> id

  type 'v insertion = {
    op : S.Dialect.op;
    outputs : 'v target list;
    from : Node_id.t list;
  }

  type placement = Inherit | New_group of string option

  type 'v replacement = {
    remove : Node_id.Set.t;
    insert : 'v insertion list;
    placement : placement;
    tensors : Tensor_sig.t list;
    subst : ('v edit_edge * 'v edit_edge) list;
    value_claims : ('v source * 'v target * Correspondence.relation) list;
    constants : ('v target * Tensor.packed) list;
    provenance : ('v source list * 'v target) list;
  }

  let empty_replacement =
    {
      remove = Node_id.Set.empty;
      insert = [];
      placement = Inherit;
      tensors = [];
      subst = [];
      value_claims = [];
      constants = [];
      provenance = [];
    }

  type error = [ `Unknown_source_edge of Tensor_id.t ]

  let pp_error fmt : [< error ] -> unit = function
    | `Unknown_source_edge id ->
        Fmt.pf fmt "@[<h>%a is not an edge of the graph being rewritten@]"
          Tensor_id.pp id

  (* State is the supply plus the replacements emitted so far, in order, plus the
     version [existing] resolves against. *)
  type 'v state = {
    ids : Id_supply.t;
    rev : 'v replacement list;
    snapshot : 'v Snap.t;
  }

  type ('v, 'a) t = 'v state -> ('a * 'v state, error) Err.t

  let return x s = Err.return (x, s)
  let ( let* ) m f s = match m s with Ok (x, s) -> f x s | Error e -> Error e

  let ( let+ ) m f s =
    match m s with Ok (x, s) -> Err.return (f x, s) | Error e -> Error e

  let all f l =
    let m =
      List.fold_left
        (fun acc x ->
          let* acc = acc in
          let+ y = f x in
          y :: acc)
        (return []) l
    in
    let+ xs = m in
    List.rev xs

  let f32 = Payload.Fmt Payload.F32

  let existing id s =
    match Snap.edge s.snapshot id with
    | Some _ -> Err.return (id, s)
    | None -> Err.fail (`Unknown_source_edge id)

  let fresh ?(fmt = f32) ?quant shape s =
    let id, ids = Id_supply.tensor s.ids in
    let sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt ?quant () in
    (* The signature rides on the next emitted replacement, which is where a
       defined edge has to be declared anyway. *)
    let rev =
      match s.rev with
      | [] -> [ { empty_replacement with tensors = [ sg ] } ]
      | r :: rest -> { r with tensors = r.tensors @ [ sg ] } :: rest
    in
    Err.return (id, { s with ids; rev })

  let emit r s =
    (* [fresh] may have parked signatures in a placeholder; fold them in. *)
    match s.rev with
    | placeholder :: rest
      when placeholder.remove = Node_id.Set.empty
           && placeholder.insert = [] && placeholder.tensors <> [] ->
        Err.return
          ( (),
            {
              s with
              rev = { r with tensors = placeholder.tensors @ r.tensors } :: rest;
            } )
    | _ -> Err.return ((), { s with rev = r :: s.rev })

  let run m snapshot ids =
    let open Err.Syntax in
    let+ x, s = m { ids; rev = []; snapshot } in
    (x, List.rev s.rev, s.ids)

  (* ---- smart constructors -------------------------------------------------- *)

  let trim ~remove ~tie =
    emit
      {
        empty_replacement with
        remove = Node_id.Set.of_list remove;
        subst = List.map (fun (from, onto) -> (Old from, Old onto)) tie;
        value_claims =
          List.map
            (fun (from, onto) ->
              (from, Preserved onto, Correspondence.Identical))
            tie;
      }

  let replace ~remove ~insert ?(tensors = []) ?(subst = []) ?(claims = [])
      ?(placement = Inherit) () =
    (* A source output the insertions take over keeps its id while changing
       definition, which [apply] requires a claim for; add the [Identical]
       self-claim unless the recipe already speaks about that edge.

       "Already speaks about" has to mean EITHER side of a claim, not just the
       source. A value-changing rewrite substitutes the other way round — old
       output onto a fresh id, claimed [(old, fresh, Equivalent)] — and inventing
       a self-claim for [fresh] would name an edge that does not exist in the
       source graph at all, which the map's endpoint checks rightly reject. *)
    let mentioned id =
      List.exists
        (fun (src, dst, _) ->
          Tensor_id.equal (raw_source src) id
          || Tensor_id.equal (raw_target dst) id)
        claims
    in
    (* Only a substitution ONTO an existing edge can need a self-claim: a fresh
       target has no source-side counterpart to claim anything about. *)
    let self_claims =
      List.filter_map
        (fun (_, onto) ->
          match onto with
          | New _ -> None
          | Old id ->
              if mentioned id then None
              else Some (id, Preserved id, Correspondence.Identical))
        subst
    in
    emit
      {
        empty_replacement with
        remove = Node_id.Set.of_list remove;
        insert;
        placement;
        tensors;
        subst;
        value_claims = claims @ self_claims;
      }

  let fold_to_constant ~node ~output ~value ~sources =
    emit
      {
        empty_replacement with
        remove = Node_id.Set.singleton node;
        constants = [ (Preserved output, value) ];
        (* Same tensor, computed earlier: the id is kept, so the claim is a
           self-claim rather than a rename. *)
        value_claims = [ (output, Preserved output, Correspondence.Identical) ];
        provenance = [ (sources, Preserved output) ];
      }

  (* ---- printing ------------------------------------------------------------ *)

  let pp_replacement fmt r =
    let pp_ids pp_id fmt l =
      Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_id) fmt l
    in
    (* Inserted nodes print as [+0], [+1]: they have no id until [apply] stamps
       one, and placeholders keep a golden stable when unrelated ids shift. *)
    let pp_insertion fmt (i, (ins : 'v insertion)) =
      Fmt.pf fmt "@[<hv 2>+%d: %a =@ %a@ from=%a@]" i (pp_ids Tensor_id.pp)
        (List.map raw_target ins.outputs)
        (S.Dialect.pp_op Tensor_id.pp)
        ins.op (pp_ids Node_id.pp) ins.from
    in
    let pp_claim fmt (src, dst, rel) =
      Fmt.pf fmt "@[<h>%a -> %a %a@]" Tensor_id.pp (raw_source src) Tensor_id.pp
        (raw_target dst) Correspondence.pp_relation rel
    in
    let pp_subst fmt (from, onto) =
      Fmt.pf fmt "@[<h>%a := %a@]" Tensor_id.pp (raw_edit_edge from)
        Tensor_id.pp (raw_edit_edge onto)
    in
    let pp_prov fmt (sources, dst) =
      Fmt.pf fmt "@[<h>%a -> %a@]" (pp_ids Tensor_id.pp)
        (List.map raw_source sources)
        Tensor_id.pp (raw_target dst)
    in
    let section name pp fmt = function
      | [] -> ()
      | l -> Fmt.pf fmt "@,@[<v 2>%s:@,%a@]" name Fmt.(list ~sep:cut pp) l
    in
    (* Sorted by the substituted id, as the [Tensor_id.Map] this used to be
       printed in order. *)
    let subst =
      List.sort
        (fun (a, _) (b, _) ->
          Tensor_id.compare (raw_edit_edge a) (raw_edit_edge b))
        r.subst
    in
    Fmt.pf fmt "@[<v>@[<h>remove: %a@]%a%a%a%a%a@]" (pp_ids Node_id.pp)
      (Node_id.Set.elements r.remove)
      (section "insert" pp_insertion)
      (List.mapi (fun i x -> (i, x)) r.insert)
      (section "subst" pp_subst) subst
      (section "claims" pp_claim)
      r.value_claims
      (section "constants" (fun fmt (id, _) ->
           Fmt.pf fmt "@[<h>%a = <payload>@]" Tensor_id.pp (raw_target id)))
      r.constants
      (section "provenance" pp_prov)
      r.provenance
end

include Make (Native_side)
