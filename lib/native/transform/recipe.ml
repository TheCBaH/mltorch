(* See recipe.mli. *)

open Graph_ir

type insertion = { op : op; outputs : Tensor_id.t list; from : Node_id.t list }
type placement = Inherit | New_group of string option

type replacement = {
  remove : Node_id.Set.t;
  insert : insertion list;
  placement : placement;
  tensors : Tensor_sig.t list;
  subst : Tensor_id.t Tensor_id.Map.t;
  value_claims : (Tensor_id.t * Tensor_id.t * Correspondence.relation) list;
  constants : (Tensor_id.t * Tensor.packed) list;
  provenance : (Tensor_id.t list * Tensor_id.t) list;
}

let empty_replacement =
  {
    remove = Node_id.Set.empty;
    insert = [];
    placement = Inherit;
    tensors = [];
    subst = Tensor_id.Map.empty;
    value_claims = [];
    constants = [];
    provenance = [];
  }

(* State is the supply plus the replacements emitted so far, in order. *)
type state = { ids : Id_supply.t; rev : replacement list }
type 'a t = state -> 'a * state

let return x s = (x, s)

let ( let* ) m f s =
  let x, s = m s in
  f x s

let ( let+ ) m f s =
  let x, s = m s in
  (f x, s)

let f32 = Payload.Fmt Payload.F32

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
  (id, { ids; rev })

let emit r s =
  (* [fresh] may have parked signatures in a placeholder; fold them in. *)
  match s.rev with
  | placeholder :: rest
    when placeholder.remove = Node_id.Set.empty
         && placeholder.insert = [] && placeholder.tensors <> [] ->
      ( (),
        {
          s with
          rev = { r with tensors = placeholder.tensors @ r.tensors } :: rest;
        } )
  | _ -> ((), { s with rev = r :: s.rev })

let run m ids =
  let x, s = m { ids; rev = [] } in
  (x, List.rev s.rev, s.ids)

(* ---- smart constructors -------------------------------------------------- *)

let map_of_list l =
  List.fold_left (fun m (k, v) -> Tensor_id.Map.add k v m) Tensor_id.Map.empty l

let trim ~remove ~tie =
  emit
    {
      empty_replacement with
      remove = Node_id.Set.of_list remove;
      subst = map_of_list tie;
      value_claims =
        List.map
          (fun (from, onto) -> (from, onto, Correspondence.Identical))
          tie;
    }

let replace ~remove ~insert ?(tensors = []) ?(subst = []) ?(claims = [])
    ?(placement = Inherit) () =
  (* Every source output the insertions take over keeps its id while changing
     definition, which [apply] requires a claim for. Add the [Identical]
     self-claim for each unless the caller said otherwise — a rewrite that
     changes the value gives the output a fresh id and claims the pair itself. *)
  let claimed id =
    List.exists (fun (src, _, _) -> Tensor_id.equal src id) claims
  in
  let self_claims =
    List.filter_map
      (fun (_, onto) ->
        if claimed onto then None
        else Some (onto, onto, Correspondence.Identical))
      subst
  in
  emit
    {
      empty_replacement with
      remove = Node_id.Set.of_list remove;
      insert;
      placement;
      tensors;
      subst = map_of_list subst;
      value_claims = claims @ self_claims;
    }

let fold_to_constant ~node ~output ~value ~sources =
  emit
    {
      empty_replacement with
      remove = Node_id.Set.singleton node;
      constants = [ (output, value) ];
      (* Same tensor, computed earlier: the id is kept, so the claim is a
         self-claim rather than a rename. *)
      value_claims = [ (output, output, Correspondence.Identical) ];
      provenance = [ (sources, output) ];
    }

(* ---- printing ------------------------------------------------------------ *)

let pp_replacement fmt r =
  let pp_ids pp_id fmt l = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_id) fmt l in
  (* Inserted nodes print as [+0], [+1]: they have no id until [apply] stamps
     one, and placeholders keep a golden stable when unrelated ids shift. *)
  let pp_insertion fmt (i, (ins : insertion)) =
    Fmt.pf fmt "@[<hv 2>+%d: %a =@ %a@ from=%a@]" i (pp_ids Tensor_id.pp)
      ins.outputs
      (Graph_ir.pp_op_with ~pp_ref:Tensor_id.pp)
      ins.op (pp_ids Node_id.pp) ins.from
  in
  let pp_claim fmt (src, dst, rel) =
    Fmt.pf fmt "@[<h>%a -> %a %a@]" Tensor_id.pp src Tensor_id.pp dst
      Correspondence.pp_relation rel
  in
  let pp_subst fmt (from, onto) =
    Fmt.pf fmt "@[<h>%a := %a@]" Tensor_id.pp from Tensor_id.pp onto
  in
  let pp_prov fmt (sources, dst) =
    Fmt.pf fmt "@[<h>%a -> %a@]" (pp_ids Tensor_id.pp) sources Tensor_id.pp dst
  in
  let section name pp fmt = function
    | [] -> ()
    | l -> Fmt.pf fmt "@,@[<v 2>%s:@,%a@]" name Fmt.(list ~sep:cut pp) l
  in
  Fmt.pf fmt "@[<v>@[<h>remove: %a@]%a%a%a%a%a@]" (pp_ids Node_id.pp)
    (Node_id.Set.elements r.remove)
    (section "insert" pp_insertion)
    (List.mapi (fun i x -> (i, x)) r.insert)
    (section "subst" pp_subst)
    (Tensor_id.Map.bindings r.subst)
    (section "claims" pp_claim)
    r.value_claims
    (section "constants" (fun fmt (id, _) ->
         Fmt.pf fmt "@[<h>%a = <payload>@]" Tensor_id.pp id))
    r.constants
    (section "provenance" pp_prov)
    r.provenance
