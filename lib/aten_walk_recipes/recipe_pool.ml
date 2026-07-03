(* Max/avg-pool2d-family walk recipe: a single spatial-window over one tensor,
   no weights or groups. Dilation is fixed at 1 (the native pool doesn't model
   dilation) and pooling additionally requires 2*pad <= kernel. *)

type t = {
  kernel_h : int;
  kernel_w : int;
  stride_h : int;
  stride_w : int;
  pad_h : int;
  pad_w : int;
  n : int;
  c : int;
  input_h : int;
  input_w : int;
}

(* Clamp pad to <= kernel/2, then grow each input for a >=1 output (dilation 1). *)
let cascade c =
  let pad_h = Window_math.clamp_pad_pool ~pad:c.pad_h ~kernel:c.kernel_h in
  let pad_w = Window_math.clamp_pad_pool ~pad:c.pad_w ~kernel:c.kernel_w in
  let input_h =
    Window_math.grow_input ~in_size:c.input_h ~pad:pad_h ~kernel:c.kernel_h
      ~dilation:1
  in
  let input_w =
    Window_math.grow_input ~in_size:c.input_w ~pad:pad_w ~kernel:c.kernel_w
      ~dilation:1
  in
  { c with pad_h; pad_w; input_h; input_w }

let self_shape c = [ c.n; c.c; c.input_h; c.input_w ]
let kernel_size c = [ c.kernel_h; c.kernel_w ]
let strides c = [ c.stride_h; c.stride_w ]
let pads c = [ c.pad_h; c.pad_w ]
let dilation _c = [ 1; 1 ]
let ceil_mode _c = false

let axes ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h ~pad_w ~n ~c ~input_h
    ~input_w =
  Walk.
    [
      field_axis "kernel_h" kernel_h (fun c v -> { c with kernel_h = v });
      field_axis "kernel_w" kernel_w (fun c v -> { c with kernel_w = v });
      field_axis "stride_h" stride_h (fun c v -> { c with stride_h = v });
      field_axis "stride_w" stride_w (fun c v -> { c with stride_w = v });
      field_axis "pad_h" pad_h (fun c v -> { c with pad_h = v });
      field_axis "pad_w" pad_w (fun c v -> { c with pad_w = v });
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "input_h" input_h (fun c v -> { c with input_h = v });
      field_axis "input_w" input_w (fun c v -> { c with input_w = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{kernel=%dx%d stride=%dx%d pad=%dx%d n=%d c=%d H=%d W=%d}"
    c.kernel_h c.kernel_w c.stride_h c.stride_w c.pad_h c.pad_w c.n c.c
    c.input_h c.input_w
