(* pad.default walk recipe. SymInt[] pad has no schema default, so this op never
   reaches the generated Default tier (op3-impl.md F5) and the recipe is the
   only walk it can have.

   ONE correlated axis of whole configurations, never independent pad/mode/value
   axes. Two separate reasons, and both are fatal on their own:

   - mode and value are not independent. ATen refuses a non-constant mode
     carrying a value, so an independent [value] axis would spend most of its
     draws on configurations ATen rejects -- which report as skips and read as
     coverage.
   - reflect's amounts are bounded by the extent they mirror ((pairs, rank) must
     be (2, 3..4) for reflect2d, and each pad must be strictly below its axis),
     so a pad list drawn independently of the shape is invalid the moment the
     shape axis moves.

   The second is handled without a [cascade]: [pads] DERIVES the reflect amounts
   from the shape currently drawn, so the correlation is by construction rather
   than repaired after the fact. [cascade] is therefore the identity, unlike
   Recipe_view's.

   Rank is fixed at 4. That is not laziness about rank coverage -- ATen
   dispatches reflect per rank and accepts two pairs only at rank 3 or 4, so a
   rank axis would have to be correlated with the pair count as well, and the
   rank-relative decoding it would exercise is already covered by the bridge and
   importer fixtures against every rank ATen accepts. *)

type pattern =
  | Const_w  (** one pair, asymmetric, explicit non-zero fill *)
  | Const_w_no_value  (** the same pair with [value] ABSENT: ATen fills 0 *)
  | Const_hw  (** two pairs, so the pair-reversal is observable *)
  | Crop_h  (** a negative pair: ATen crops, and nothing is synthesized *)
  | Crop_and_pad  (** crop one axis while padding the other *)
  | Reflect_hw  (** symmetric mirror on both axes *)
  | Reflect_asym_w  (** asymmetric mirror, zero pair on the other axis *)

type t = { n : int; c : int; h : int; w : int; pattern : pattern }

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]

(* Strictly below the extent it mirrors, which is ATen's own rule for reflect.
   Derived from the shape in the config rather than stored, so a step on [h] or
   [w] cannot leave a stale amount behind. *)
let mirror ~extent want = min want (extent - 1)

(* PyTorch's own order: innermost dimension FIRST, one (before, after) pair per
   padded dimension. Written that way here deliberately -- the reversal is the
   importers' job, and a recipe that pre-un-reversed it would be testing the
   convention against itself. *)
let pads c =
  match c.pattern with
  | Const_w -> [ 1; 2 ]
  | Const_w_no_value -> [ 2; 1 ]
  | Const_hw -> [ 1; 2; 3; 1 ]
  | Crop_h -> [ 0; 0; -1; -1 ]
  | Crop_and_pad -> [ -1; 0; 2; 1 ]
  | Reflect_hw ->
      let mw = mirror ~extent:c.w 1 and mh = mirror ~extent:c.h 1 in
      [ mw; mw; mh; mh ]
  | Reflect_asym_w ->
      let mw = mirror ~extent:c.w 2 in
      [ mw; mirror ~extent:c.w 1; 0; 0 ]

let mode c =
  match c.pattern with
  | Reflect_hw | Reflect_asym_w -> "reflect"
  | Const_w | Const_w_no_value | Const_hw | Crop_h | Crop_and_pad -> "constant"

(* Non-zero, and not f32-exact, so a fill that was dropped or narrowed on the
   way through is a value difference rather than a coincidence. [None] for
   reflect because ATen refuses a value there, and for one constant case
   because absent must mean 0 and that is a rule worth exercising. *)
let value c =
  match c.pattern with
  | Const_w -> Some (Aten_spec.Float32.to_f32 0.1)
  | Const_hw -> Some (Aten_spec.Float32.to_f32 (-2.5))
  | Crop_h -> Some (Aten_spec.Float32.to_f32 7.)
  | Crop_and_pad -> Some (Aten_spec.Float32.to_f32 0.1)
  | Const_w_no_value | Reflect_hw | Reflect_asym_w -> None

let all_patterns =
  [
    Const_w;
    Const_w_no_value;
    Const_hw;
    Crop_h;
    Crop_and_pad;
    Reflect_hw;
    Reflect_asym_w;
  ]

let axes ~n ~c ~h ~w ~pattern =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "pattern" pattern (fun c v -> { c with pattern = v });
    ]

let pp_pattern = function
  | Const_w -> "const_w"
  | Const_w_no_value -> "const_w_no_value"
  | Const_hw -> "const_hw"
  | Crop_h -> "crop_h"
  | Crop_and_pad -> "crop_and_pad"
  | Reflect_hw -> "reflect_hw"
  | Reflect_asym_w -> "reflect_asym_w"

let pp ppf c =
  let ints l = String.concat "," (List.map string_of_int l) in
  Format.fprintf ppf "{shape=[%s] pattern=%s pad=[%s] mode=%s value=%s}"
    (ints (self_shape c))
    (pp_pattern c.pattern)
    (ints (pads c))
    (mode c)
    (match value c with None -> "none" | Some v -> Printf.sprintf "%g" v)
