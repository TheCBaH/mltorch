(* The product of extents formed inside one tensor, bounded where it is formed.

   These products are sub-products of a tensor's extents, so a graph whose
   tensors are bounded by [Kernel.Limits.Hard.numel] never exceeds it. That was
   an assumption about where the factors came from — the validated graph view
   does not enforce it — and js_of_ocaml's [int] is 32 bits, so a product that
   does not fit would wrap silently. [bounded] checks at the point of use
   instead: [None] means the product reaches the numel ceiling, and every
   caller declines (a lowering recognizer falls back to the ordinary path). *)

val bounded : Dim.extent Dim.t list -> Dim.extent Dim.t option
