(* Axis permutation: a full 6D rearrangement with data copy. Used by the bridge
   to relayout tensors between ATen's positional right-alignment and native
   semantic axis assignments (e.g. NCHW -> NHWC for conv/pool inputs/outputs).

   A permutation is an association list [(out_ax, in_ax)] covering all 6 axes
   exactly once on each side: output axis [out_ax] gets the extent and data from
   input axis [in_ax].  The type alias keeps call sites readable — the direction
   is always output-to-input. *)

module Permute = struct
  (* [(out_axis, in_axis)] — which input axis feeds each output axis. *)
  type perm = (Axis.t * Axis.t) list

  (* output_shape[out_ax] = x_shape[in_ax] for each (out_ax, in_ax) pair.
     Axes absent from [perm] default to extent 1 (never happens for a full perm). *)
  let output_shape ~(x_shape : Vec6.shape) (perm : perm) =
    List.fold_left
      (fun s (out_ax, in_ax) -> Vec6.set s out_ax (Vec6.get x_shape in_ax))
      (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
      perm

  module Compute (S : Semantics.SEMANTICS) = struct
    (* At each output coord [out], read the input at the coordinate produced by
       the inverse permutation: for each input axis [in_ax], use the output
       coordinate of the output axis that maps to it.
       Raises [Not_found] if [perm] is not a bijection over all 6 axes. *)
    let pixel perm ~x (out : Axis.t -> Semantics.position S.index) =
      let inv = List.map (fun (oax, iax) -> (iax, oax)) perm in
      S.load x (fun in_ax -> out (List.assoc in_ax inv))
  end
end
