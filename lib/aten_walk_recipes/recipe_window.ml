(* [Walk_core.Window_math] over the plain ints an ATen recipe's config holds:
   they are the values written into ATen call arguments, so the guarded types
   are entered here and left again, once, at this boundary. *)

let grow_input ~in_size ~pad ~kernel ~dilation =
  (Walk_core.Window_math.grow_input ~in_size:(Core.Dim.extent in_size)
     ~pad:(Core.Geometry.Nonneg.of_int pad)
     ~kernel:(Core.Dim.extent kernel)
     ~dilation:(Core.Geometry.Pos.of_int dilation)
    :> int)

let clamp_pad_pool ~pad ~kernel =
  (Walk_core.Window_math.clamp_pad_pool
     ~pad:(Core.Geometry.Nonneg.of_int pad)
     ~kernel:(Core.Dim.extent kernel)
    :> int)

let round_up_multiple ~n ~m =
  (Walk_core.Window_math.round_up_multiple ~n:(Core.Dim.extent n)
     ~m:(Core.Dim.extent m)
    :> int)
