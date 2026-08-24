(* bmm.default walk recipe: batch/row/contract/col drawn independently -- no
   correlation is needed since [self]/[mat2]'s shapes are already jointly
   determined by the four scalars ([Native.Matmul.Bmm]'s own contract, see
   .ai/native_compute_design.md §2), unlike e.g. [Recipe_transpose] where a
   dim pair is only valid for SOME ranks. *)

type t = { batch : int; n : int; m : int; p : int }

let cascade c = c
let self_shape c = [ c.batch; c.n; c.m ]
let mat2_shape c = [ c.batch; c.m; c.p ]

let axes ~batch ~n ~m ~p =
  Walk.
    [
      field_axis "batch" batch (fun c v -> { c with batch = v });
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "m" m (fun c v -> { c with m = v });
      field_axis "p" p (fun c v -> { c with p = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{self=[%d,%d,%d] mat2=[%d,%d,%d]}" c.batch c.n c.m c.batch
    c.m c.p
