(* Fully-connected / `aten.addmm` layer. Output[n,...,oc] reduces over
   in_features at the same (n,t,d,h,w) position — no spatial kernel, so this
   is [Conv2d]'s [cin] reduction alone, without the kh/kw window (resnet18's
   FC layer runs on an already globally-pooled, spatially-1x1 tensor, so a
   spatial kernel would be vacuous here anyway). Weight is laid out
   [Out,1,1,1,1,In] (matching [Conv2d]'s [Cout,1,1,Kh,Kw,Cin] minus the kernel
   axes); bias is [1,1,1,1,1,Out]. See .ai/native_compute_design.md §2. *)

module Linear = struct
  type params = { in_features : Dim.extent Dim.t }

  (* N/T/D/H/W pass through from [x_shape] (no spatial reduction); only C
     changes, to [weight_shape]'s [Out]. Extent-space, no [:> int] round-trips.
     See .ai/native_compute_design.md §2b. *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape) =
    Vec6.set x_shape Axis.C (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~x ~weight ~bias
        (out : Axis.t -> Semantics.position S.index) =
      let oc = out Axis.C in
      let acc =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent p.in_features) (fun ic ->
            let x_idx a = match a with Axis.C -> ic | _ -> out a in
            let w_idx a =
              match a with Axis.N -> oc | Axis.C -> ic | _ -> S.index_zero
            in
            S.mul (S.load x x_idx) (S.load weight w_idx))
      in
      S.add acc
        (S.load bias (fun a -> match a with Axis.C -> oc | _ -> S.index_zero))
  end
end
