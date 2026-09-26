(* See extent_product.mli. *)

let bounded extents =
  Result.to_option
    (Err.payload (Dim.product_bounded ~limit:Kernel.Limits.Hard.numel extents))
