(* Transport a permutation through a [keepdim=true] [Mean], the axis-covariant
   counterpart to [Sink_permute]'s axis-invariant elementwise identity. See
   .ai/native_transform_design.md §12f.

   [Mean] reduces over NAMED axes, so it does not commute with an input
   permutation the way an elementwise op does — [Sink_permute] deliberately
   excludes it (§12d). But the reduction dimensions can be carried through the
   permutation instead, which is a different, and still exact, identity. For
   [P] an output-to-input permutation and [D] the (ordered) reduction axes:

     Mean_D(P(x)) = P(Mean_{P(D)}(x))          (keepdim=true only)

   [P(D)] is [List.map (Permute.Permute.lookup P) D] — [Reduce.Mean.map_dims]
   applied with that lookup. [keepdim=true] is what makes the induced output
   permutation exactly [P] again (reduced axes collapse in place rather than
   being repacked by [Mean.kept_map]), so the rewrite needs no separate
   output-permutation derivation the way [keepdim=false] eventually will. *)

open Graph_ir

let as_mean = function Mean (m : Reduce.Mean.t) -> Some m | _ -> None

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

(* [packed_fmt] is an existential over a GADT and has no usable structural
   order, so compare by format name, same as [Bypass_permute.precision_equal]. *)
let precision_equal (a : Tensor_sig.t) (b : Tensor_sig.t) =
  let fmt_key (Payload.Fmt f) = Payload.fmt_name f in
  String.equal (fmt_key a.Tensor_sig.fmt) (fmt_key b.Tensor_sig.fmt)
  && Stdlib.( = ) a.Tensor_sig.quant b.Tensor_sig.quant

module Match = struct
  type t = {
    mean_node : Node_id.t;
    permute_node : Node_id.t;
    perm : Permute.Permute.perm;
    mapped_params : Reduce.Mean.params;
    mapped_shape : Vec6.shape; (* the transported Mean's own output shape *)
    x : Tensor_id.t;
    out : Tensor_id.t;
    z : Tensor_sig.t;
  }
end

let pattern anchor =
  let open Pattern in
  let* (mean : Reduce.Mean.t), (mean_node : node) = def anchor as_mean in
  let* () = guard mean.Reduce.Mean.params.Reduce.Mean.keepdim in
  let* () = interior mean.Reduce.Mean.x in
  let* (p : Permute.Permute.t), (permute_node : node) =
    def mean.Reduce.Mean.x as_permute
  in
  let* (x_sig : Tensor_sig.t) = sig_of p.x in
  let* (y_sig : Tensor_sig.t) = sig_of mean.Reduce.Mean.x in
  let* () = guard (precision_equal x_sig y_sig) in
  let mapped_params =
    Reduce.Mean.map_dims (Permute.Permute.lookup p.perm) mean.Reduce.Mean.params
  in
  let* mapped_shape =
    match
      Reduce.Mean.output_shape ~x_shape:x_sig.Tensor_sig.shape mapped_params
    with
    | Ok shape -> return shape
    | Error _ -> fail Rejected
  in
  let* final_shape =
    match Permute.Permute.output_shape ~x_shape:mapped_shape p.perm with
    | Ok shape -> return shape
    | Error _ -> fail Rejected
  in
  let* (z_sig : Tensor_sig.t) = sig_of anchor in
  let+ () = guard (Stdlib.( = ) final_shape z_sig.Tensor_sig.shape) in
  {
    Match.mean_node = mean_node.Node.id;
    permute_node = permute_node.Node.id;
    perm = p.perm;
    mapped_params;
    mapped_shape;
    x = p.x;
    out = anchor;
    z = z_sig;
  }

let build (m : Match.t) _region =
  let open Recipe in
  let* pre =
    fresh ~fmt:m.Match.z.Tensor_sig.fmt ?quant:m.Match.z.Tensor_sig.quant
      m.Match.mapped_shape
  in
  let* out = existing m.Match.out in
  replace
    ~remove:[ m.Match.mean_node; m.Match.permute_node ]
    ~insert:
      [
        {
          Recipe.op =
            Mean { Reduce.Mean.params = m.Match.mapped_params; x = m.Match.x };
          outputs = [ Fresh pre ];
          from = [ m.Match.mean_node ];
        };
        {
          Recipe.op = Permute { perm = m.Match.perm; x = raw_fresh pre };
          outputs = [ Preserved out ];
          from = [ m.Match.permute_node ];
        };
      ]
    ~claims:[ (out, Preserved out, Correspondence.Identical) ]
    ()

let pass =
  Pass.of_pattern ~name:"sink_permute_mean" ~pattern ~build:{ Pass.build }
