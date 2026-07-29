(* Rewrite a contiguous reshape that only relabels axes into the permute it
   really is. See .ai/native_transform_design.md §12.

   A [Reshape] sends element k of the row-major input to element k of the
   row-major output. Unit axes contribute nothing to a row-major offset, so when
   the non-unit extents of the two shapes agree *in axis order* the mapping is
   exactly the bijection carrying the i-th non-unit input axis onto the i-th
   non-unit output axis. Anything else — a genuine flatten or split — mixes
   extents and is not expressible as a permutation of the six axes.

   Worth doing because a permute composes: it fuses with its neighbours
   ([Chain_permute]) and cancels against them ([Trim_permute]), where a reshape
   is opaque to both. *)

open Graph_ir

type match_ = {
  node : Node_id.t;
  out : Tensor_id.t; (* the permute takes this id over *)
  perm : Permute.Permute.perm;
  x : Tensor_id.t;
}

let as_reshape = function
  | Reshape (r : Reshape.Reshape.t) -> Some r
  | _ -> None

let non_unit shape =
  List.filter
    (fun axis -> not (Dim.equal (Vec6.get shape axis) Dim.one))
    Axis.all

(* [Some perm] when the reshape is pure relabelling. Leftover unit axes are
   paired in canonical order: being unit they compute the same tensor whichever
   way they are matched, and a fixed order keeps the result deterministic. *)
let relabelling ~from_shape ~to_shape =
  let src = non_unit from_shape and dst = non_unit to_shape in
  let agree =
    List.length src = List.length dst
    && List.for_all2
         (fun a b -> Dim.equal (Vec6.get from_shape a) (Vec6.get to_shape b))
         src dst
  in
  if not agree then None
  else
    let rest taken =
      List.filter (fun axis -> not (List.mem axis taken)) Axis.all
    in
    let pairs = List.combine dst src @ List.combine (rest dst) (rest src) in
    Some (Permute.Permute.of_fn (Permute.Permute.lookup pairs))

let pattern anchor =
  let open Pattern in
  let* (r : Reshape.Reshape.t), (node : node) = def anchor as_reshape in
  let* (from : Tensor_sig.t) = sig_of r.x in
  match relabelling ~from_shape:from.shape ~to_shape:r.params.shape with
  | None -> fail Rejected
  | Some perm -> return { node = node.Node.id; out = anchor; perm; x = r.x }

(* Same tensor, same shape, different definition — so the id is kept and the
   self-claim states why that is legal. *)
let build m _region =
  let open Recipe in
  let* out = existing m.out in
  replace ~remove:[ m.node ]
    ~insert:
      [
        {
          Recipe.op = Permute { perm = m.perm; x = m.x };
          outputs = [ Preserved out ];
          from = [ m.node ];
        };
      ]
    ~claims:[ (out, Preserved out, Correspondence.Identical) ]
    ()

let pass =
  Pass.of_pattern ~name:"reshape_to_permute" ~pattern ~build:{ Pass.build }
