(* Shared helpers for the Direct-evaluator tests split out of what was
   compute_test.ml. Coordinate readers ([chan]/[row]/[col]),
   [s1c] (a single-channel-only shape), [eval_tensor] (thread a [Shape_error]
   through evaluation), and [pp_result] (print an evaluated tensor or its
   error) are used across nearly every split file; anything narrower stays
   local to the file that needs it (e.g. [conv_axis]/[conv_params] in
   conv_test.ml, [bat] in sdpa_test.ml). *)

let chan c = Dim.to_int (Vec6.get c Axis.C)
let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)
let bat c = Dim.to_int (Vec6.get c Axis.D)
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let eval_tensor shape_result pixel =
  let open Err.Syntax in
  let* out_shape = shape_result in
  Err.return (Schedule.evaluate out_shape pixel)

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:Shape_error.pp

(* A coordinate-coded tensor: element (h, w) is 10*h + w, so every value names
   the position it came from and a wrong source coordinate is legible rather
   than merely unequal. Shared by pad_test.ml and slice_select_test.ml. *)
let coord_hw h w =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1 in
  ( shape,
    Tensor.materialize shape (fun c -> float_of_int ((10 * row c) + col c)) )

(* [Tensor.pp] truncates after 8 elements and prints one flat list, which is the
   wrong shape of evidence for a pad or slice: judged by WHERE each value
   landed. Print the H x W grid in full, so a mirrored edge or a swapped
   before/after is legible rather than merely unequal. *)
let pp_grid shape ppf tensor =
  let h = Dim.to_int (Vec6.get shape Axis.H)
  and w = Dim.to_int (Vec6.get shape Axis.W) in
  Format.fprintf ppf "%a@," Vec6.pp_shape shape;
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:i ~w:j ~c:0 in
      Format.fprintf ppf "%s%g"
        (if j = 0 then "" else " ")
        (Tensor.read tensor c)
    done;
    Format.fprintf ppf "@,"
  done
