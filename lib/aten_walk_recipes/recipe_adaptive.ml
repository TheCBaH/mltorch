(* adaptive_avg_pool2d walk recipe: a 4D tensor pooled to an explicit output
   spatial size. Adaptive pooling is valid for any positive input and output
   sizes (output may exceed input), so [cascade] is identity. *)

type t = {
  n : int;
  c : int;
  input_h : int;
  input_w : int;
  out_h : int;
  out_w : int;
}

let cascade c = c
let self_shape c = [ c.n; c.c; c.input_h; c.input_w ]
let output_size c = [ c.out_h; c.out_w ]

let axes ~n ~c ~input_h ~input_w ~out_h ~out_w =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "input_h" input_h (fun c v -> { c with input_h = v });
      field_axis "input_w" input_w (fun c v -> { c with input_w = v });
      field_axis "out_h" out_h (fun c v -> { c with out_h = v });
      field_axis "out_w" out_w (fun c v -> { c with out_w = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] output_size=[%d,%d]}" c.n c.c
    c.input_h c.input_w c.out_h c.out_w
