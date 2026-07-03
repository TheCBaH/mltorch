(* conv2d.padding walk recipe: a grouped 2D convolution whose padding is a string
   mode ("valid" | "same") rather than explicit ints. PyTorch forbids "same"
   padding with a stride != 1, so the cascade demonstrates a genuine cross-arg
   constraint: choosing "same" forces the strides back to 1. *)

type t = {
  kernel_h : int;
  kernel_w : int;
  stride_h : int;
  stride_w : int;
  dilation_h : int;
  dilation_w : int;
  groups : int;
  in_channels : int;
  out_channels : int;
  n : int;
  input_h : int;
  input_w : int;
  padding : string; (* "valid" | "same" *)
}

(* "same" requires unit stride; channels must be multiples of groups; each input
   must hold the effective kernel (a safe lower bound for both modes). *)
let cascade c =
  let stride_h, stride_w =
    if c.padding = "same" then (1, 1) else (c.stride_h, c.stride_w)
  in
  let in_channels =
    Window_math.round_up_multiple ~n:c.in_channels ~m:c.groups
  in
  let out_channels =
    Window_math.round_up_multiple ~n:c.out_channels ~m:c.groups
  in
  let input_h =
    Window_math.grow_input ~in_size:c.input_h ~pad:0 ~kernel:c.kernel_h
      ~dilation:c.dilation_h
  in
  let input_w =
    Window_math.grow_input ~in_size:c.input_w ~pad:0 ~kernel:c.kernel_w
      ~dilation:c.dilation_w
  in
  { c with stride_h; stride_w; in_channels; out_channels; input_h; input_w }

let input_shape c = [ c.n; c.in_channels; c.input_h; c.input_w ]

let weight_shape c =
  [ c.out_channels; c.in_channels / c.groups; c.kernel_h; c.kernel_w ]

let bias_shape c = [ c.out_channels ]
let strides c = [ c.stride_h; c.stride_w ]
let dilations c = [ c.dilation_h; c.dilation_w ]
let padding c = c.padding
let groups c = c.groups

let axes ~kernel_h ~kernel_w ~stride_h ~stride_w ~dilation_h ~dilation_w ~groups
    ~in_channels ~out_channels ~n ~input_h ~input_w ~padding =
  Walk.
    [
      field_axis "kernel_h" kernel_h (fun c v -> { c with kernel_h = v });
      field_axis "kernel_w" kernel_w (fun c v -> { c with kernel_w = v });
      field_axis "stride_h" stride_h (fun c v -> { c with stride_h = v });
      field_axis "stride_w" stride_w (fun c v -> { c with stride_w = v });
      field_axis "dilation_h" dilation_h (fun c v -> { c with dilation_h = v });
      field_axis "dilation_w" dilation_w (fun c v -> { c with dilation_w = v });
      field_axis "groups" groups (fun c v -> { c with groups = v });
      field_axis "in_channels" in_channels (fun c v ->
          { c with in_channels = v });
      field_axis "out_channels" out_channels (fun c v ->
          { c with out_channels = v });
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "input_h" input_h (fun c v -> { c with input_h = v });
      field_axis "input_w" input_w (fun c v -> { c with input_w = v });
      field_axis "padding" padding (fun c v -> { c with padding = v });
    ]

let pp ppf c =
  Format.fprintf ppf
    "{kernel=%dx%d stride=%dx%d dilation=%dx%d groups=%d in_c=%d out_c=%d n=%d \
     H=%d W=%d padding=%s}"
    c.kernel_h c.kernel_w c.stride_h c.stride_w c.dilation_h c.dilation_w
    c.groups c.in_channels c.out_channels c.n c.input_h c.input_w c.padding
