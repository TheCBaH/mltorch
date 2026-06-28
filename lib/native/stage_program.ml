(* See stage_program.mli. Grounding keys the [Schedule.ground] binding by the
   signature's (globally unique) id, seeded with the graph inputs and the synthetic
   constants, then extended stage by stage in topo order. *)

open Graph_ir

module Stage = struct
  type t = { id : Tensor_id.t; sg : Tensor_sig.t; body : Expr.t }
end

type t = {
  inputs : (Tensor_id.t * Tensor_sig.t) list;
  consts : (Tensor_sig.t * float) list;
  stages : Stage.t list;
  outputs : Tensor_id.t list;
}

let pp fmt (p : t) =
  let names =
    List.map (fun (id, (s : Tensor_sig.t)) -> (id, s.name)) p.inputs
    @ List.map (fun (st : Stage.t) -> (st.id, st.sg.name)) p.stages
  in
  let name_of id =
    match List.assoc_opt id names with
    | Some n -> n
    | None -> Format.asprintf "%a" Tensor_id.pp id
  in
  let comma = String.concat ", " in
  Format.fprintf fmt "@[<v>inputs: %s@,"
    (comma (List.map (fun (_, (s : Tensor_sig.t)) -> s.name) p.inputs));
  List.iter
    (fun (st : Stage.t) ->
      Format.fprintf fmt "%s = %a@," st.sg.name Expr.pp st.body)
    p.stages;
  Format.fprintf fmt "outputs: %s@]" (comma (List.map name_of p.outputs))

(* sig.id (global) -> packed; this is what Schedule.ground's binding resolves. *)
module Sig_map = Map.Make (Int)

let ground (p : t) ~(bind : Tensor_id.t -> Tensor.packed) :
    Tensor.packed Tensor_id.Map.t =
  let seed =
    List.fold_left
      (fun m (id, (s : Tensor_sig.t)) -> Sig_map.add s.id (bind id) m)
      Sig_map.empty p.inputs
  in
  let seed =
    List.fold_left
      (fun m ((s : Tensor_sig.t), v) ->
        Sig_map.add s.id (Tensor.materialize s.shape (fun _ -> v)) m)
      seed p.consts
  in
  (* Thread the sig->tensor binding through the stages in topo order, collecting
     each stage's grounded result keyed by its edge id. *)
  let _, result =
    List.fold_left
      (fun (binds, result) (st : Stage.t) ->
        let binding (s : Tensor_sig.t) = Sig_map.find s.id binds in
        let t = Schedule.ground st.sg.shape ~binding st.body in
        (Sig_map.add st.sg.id t binds, Tensor_id.Map.add st.id t result))
      (seed, Tensor_id.Map.empty)
      p.stages
  in
  result
