(* The normalization walk recipe, shared by rms_norm.default,
   layer_norm.default and native_layer_norm.default: a rank-N input normalized
   over its trailing k axes, with optional affine operands carrying exactly
   those extents.

   The shapes are CORRELATED by construction rather than by a cascade:
   [normalized] is one list, and the input's trailing extents, the weight's
   shape and the bias's shape are all that same list. An uncorrelated walk here
   would generate specs ATen rejects, which degrade to skips -- and a suite of
   skips reads as coverage while being none.

   [leading] carries the whole leading shape as one value, so a step varies the
   input's RANK as well as its extents; [normalized] does the same for k.

   [bias] and [cudnn] belong to the layer-norm targets only, and their axes are
   OMITTED rather than defaulted to a one-element list when a target has no such
   argument. A single-valued axis is not free: the walk still spends a step on
   it and still reports it as the mutated field, which is a step that changed
   nothing dressed up as one that did. *)

type t = {
  leading : int list;
  normalized : int list;
  eps : float option;
  weight : bool;
  bias : bool;
  cudnn : bool;
}

let cascade c = c
let input_shape c = c.leading @ c.normalized
let normalized_shape c = c.normalized
let weight_shape c = c.normalized
let bias_shape c = c.normalized
let has_weight c = c.weight
let has_bias c = c.bias
let cudnn c = c.cudnn
let eps c = c.eps

(* [native_layer_norm]'s eps is required with no schema default, so its walk
   never draws [None]; this reads whatever the axis produced and supplies the
   functional overload's own default when it is absent, rather than inventing a
   third number. *)
let eps_or_default c = match c.eps with Some e -> e | None -> 1e-05

let axes ~leading ~normalized ~eps ~weight ?bias ?cudnn () =
  let opt name values setter =
    match values with
    | None -> []
    | Some vs -> [ Walk.field_axis name vs setter ]
  in
  Walk.
    [
      field_axis "leading" leading (fun c v -> { c with leading = v });
      field_axis "normalized" normalized (fun c v -> { c with normalized = v });
      field_axis "eps" eps (fun c v -> { c with eps = v });
      field_axis "weight" weight (fun c v -> { c with weight = v });
    ]
  @ opt "bias" bias (fun c v -> { c with bias = v })
  @ opt "cudnn_enable" cudnn (fun c v -> { c with cudnn = v })

(* The WHOLE config, including the fields a given target does not use. One
   recipe serves three targets, so a renderer that hid the unused fields would
   have to know which target it was printing for -- and the golden's job is to
   show what the walk actually drew. *)
let pp ppf c =
  Format.fprintf ppf
    "{input=[%s] normalized=[%s] eps=%s weight=%b bias=%b cudnn=%b}"
    (String.concat "," (List.map string_of_int (input_shape c)))
    (String.concat "," (List.map string_of_int c.normalized))
    (match c.eps with None -> "default" | Some e -> Printf.sprintf "%g" e)
    c.weight c.bias c.cudnn
