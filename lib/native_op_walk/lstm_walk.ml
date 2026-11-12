(* Lstm's random-walk config space, kept beside [Lstm_nwalk], which
   assembles it into a walk subject. It reads the op's shape rules from
   [Lstm.Lstm] and adds nothing to the op itself. *)

open Lstm.Lstm

(* Config space for the random walk: every tensor shape is DERIVED from one
   correlated record ([hidden_size]/[input_size]/[seq]/[batch] plus the
   layer/direction/bias counts), the same discipline
   Aten_walk_recipes.Recipe_lstm uses on the ATen side -- so [cascade] has
   nothing to repair. [num_layers] is capped at 3, not [L.limits]'s channel
   cap: the scan/Region machinery's cost is already linear per layer times
   [seq], and nothing about layer *count* needs the same headroom as a
   single dimension's extent. *)
module Walk (L : Walk_core.Limits.S) = struct
  type cfg = {
    hidden_size : int;
    input_size : int;
    seq : int;
    batch : int;
    num_layers : int;
    bidirectional : bool;
    bias : bool;
    batch_first : bool;
  }

  let l = L.limits

  let initial =
    {
      hidden_size = 3;
      input_size = 2;
      seq = 4;
      batch = 2;
      num_layers = 1;
      bidirectional = false;
      bias = true;
      batch_first = false;
    }

  (* [num_layers] and [bidirectional] both multiply the number of
     independent scan copies the Region program builds (one per
     layer/direction PER output ordinal, an accepted
     explicit constant-factor cost) -- a combination the per-axis [~hi]
     bounds below cannot see, since neither axis knows about the other. A
     walk step drawing [num_layers=3] together with [bidirectional=true]
     (6 layer-directions x 3 output ordinals) exceeds the Region body-size
     budget even at the smallest [hidden_size]/[seq] -- found via the
     S6/T6.4 op-sweep reseed (inserting new walks ahead of [lstm]
     alphabetically shifted its seed into this combination, which the old
     seed had never reached). Capped here rather than by tightening
     [num_layers]'s own [~hi]: a bidirectional LSTM with more than one
     layer is still a real, useful walk target, just not at
     [num_layers=3]. *)
  let cascade c =
    if c.bidirectional && c.num_layers > 1 then { c with num_layers = 1 } else c

  let directions (c : cfg) = if c.bidirectional then 2 else 1

  let params (c : cfg) : params =
    {
      hidden_size = c.hidden_size;
      input_size = c.input_size;
      batch_first = c.batch_first;
    }

  let input_shape (c : cfg) =
    let h, w = if c.batch_first then (c.batch, c.seq) else (c.seq, c.batch) in
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:c.input_size

  let state_shape (c : cfg) =
    state_shape ~p:(params c)
      ~layers:(c.num_layers * directions c)
      ~batch:c.batch

  (* Layer 0 reads the raw input width; every later layer reads the
     previous layer's [R]-wide hidden trace (the LSTM lowering analysis), matching
     [Computation.program]'s [this_layer_input_size]. *)
  let layer_input_size (c : cfg) ~layer =
    if layer = 0 then c.input_size else c.hidden_size * directions c

  let weight_ih_shape (c : cfg) ~layer =
    weight_ih_shape (params c) ~layer_input_size:(layer_input_size c ~layer)

  let weight_hh_shape (c : cfg) = weight_hh_shape (params c)
  let bias_shape (c : cfg) = bias_shape (params c)

  let axes =
    Walk_core.Walk.
      [
        int_axis "hidden_size" ~lo:1 ~hi:l.max_channels (fun c v ->
            { c with hidden_size = v });
        int_axis "input_size" ~lo:1 ~hi:l.max_channels (fun c v ->
            { c with input_size = v });
        int_axis "seq" ~lo:1 ~hi:l.max_extent (fun c v -> { c with seq = v });
        int_axis "batch" ~lo:1 ~hi:l.max_batch (fun c v -> { c with batch = v });
        int_axis "num_layers" ~lo:1 ~hi:3 (fun c v -> { c with num_layers = v });
        field_axis "bidirectional" [ true; false ] (fun c v ->
            { c with bidirectional = v });
        field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        field_axis "batch_first" [ true; false ] (fun c v ->
            { c with batch_first = v });
      ]

  let pp fmt (c : cfg) =
    Format.fprintf fmt
      "{hidden_size=%d input_size=%d seq=%d batch=%d num_layers=%d \
       bidirectional=%b bias=%b batch_first=%b}"
      c.hidden_size c.input_size c.seq c.batch c.num_layers c.bidirectional
      c.bias c.batch_first

  (* One direction's operand shapes at [layer], forward and reverse alike
     (a bidirectional layer's reverse direction reads the same width as its
     forward, the LSTM lowering analysis). [Lstm_nwalk] builds the actual graph from
     these; that needs [Graph_builder], which depends on [Graph_ir], which
     depends on [Lstm.Lstm] (the graph-IR variant), so the graph-building
     half lives here, one layer above [native]. *)
  let direction_shapes (c : cfg) ~layer =
    ( weight_ih_shape c ~layer,
      weight_hh_shape c,
      if c.bias then Some (bias_shape c, bias_shape c) else None )
end
