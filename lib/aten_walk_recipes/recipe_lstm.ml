(* The lstm.input walk recipe. Every tensor shape is DERIVED from one
   correlated record -- batch/seq/input_size/hidden_size plus the
   layer/direction/bias counts -- so [cascade] has nothing to repair, the same
   discipline [Recipe_sdpa]/[Recipe_norm] use.

   [param_shapes] mirrors the exact ordering [lstm-plan.md] §2 records: layer
   by layer, forward before reverse, and within a direction
   [weight_ih, weight_hh, [bias_ih, bias_hh]]. Layer 0 reads the raw input
   width; every later layer reads the previous layer's output, which is
   [directions * hidden_size] wide regardless of which direction is reading
   it (a bidirectional layer's output concatenates both directions). Getting
   this order or either width wrong does not merely mis-walk -- ATen's own
   [gather_params]/matrix multiplies would reject or silently mispair the
   tensors, so the recipe is the one place this contract must stay exact. *)

type t = {
  batch : int;
  seq : int;
  input_size : int;
  hidden_size : int;
  layers : int;
  bidirectional : bool;
  has_biases : bool;
  batch_first : bool;
  dropout : float;
}

let cascade c = c
let directions c = if c.bidirectional then 2 else 1

let input_shape c =
  if c.batch_first then [ c.batch; c.seq; c.input_size ]
  else [ c.seq; c.batch; c.input_size ]

let state_shape c = [ c.layers * directions c; c.batch; c.hidden_size ]
let has_biases c = c.has_biases
let num_layers c = c.layers
let dropout c = c.dropout
let bidirectional c = c.bidirectional
let batch_first c = c.batch_first

(* One direction's input width: the raw embedding for layer 0, the previous
   layer's (both-directions-wide) output for every later layer. *)
let layer_input_size c layer =
  if layer = 0 then c.input_size else c.hidden_size * directions c

let param_shapes c =
  let direction_shapes layer =
    let in_size = layer_input_size c layer in
    let core =
      [ [ 4 * c.hidden_size; in_size ]; [ 4 * c.hidden_size; c.hidden_size ] ]
    in
    if c.has_biases then core @ [ [ 4 * c.hidden_size ]; [ 4 * c.hidden_size ] ]
    else core
  in
  List.concat_map
    (fun layer ->
      List.concat_map
        (fun _ -> direction_shapes layer)
        (List.init (directions c) Fun.id))
    (List.init c.layers Fun.id)

let axes ~batch ~seq ~input_size ~hidden_size ~layers ~bidirectional ~has_biases
    ~batch_first ~dropout () =
  Walk.
    [
      field_axis "batch" batch (fun c v -> { c with batch = v });
      field_axis "seq" seq (fun c v -> { c with seq = v });
      field_axis "input_size" input_size (fun c v -> { c with input_size = v });
      field_axis "hidden_size" hidden_size (fun c v ->
          { c with hidden_size = v });
      field_axis "layers" layers (fun c v -> { c with layers = v });
      field_axis "bidirectional" bidirectional (fun c v ->
          { c with bidirectional = v });
      field_axis "has_biases" has_biases (fun c v -> { c with has_biases = v });
      field_axis "batch_first" batch_first (fun c v ->
          { c with batch_first = v });
      field_axis "dropout" dropout (fun c v -> { c with dropout = v });
    ]

let pp ppf c =
  Format.fprintf ppf
    "{batch=%d seq=%d input=%d hidden=%d layers=%d bidirectional=%b \
     has_biases=%b batch_first=%b dropout=%g}"
    c.batch c.seq c.input_size c.hidden_size c.layers c.bidirectional
    c.has_biases c.batch_first c.dropout
