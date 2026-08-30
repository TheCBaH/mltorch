(* [Recipe_pool] plus a [count_include_pad] axis -- the only pooling target
   whose divisor depends on it ([max_pool2d] has no divisor to vary). Its own
   recipe rather than a field on [Recipe_pool] itself, the same reason
   [max_pool2d]'s [dilation] stays a hardcoded constant there instead of a
   dead axis on every other pooling walk. *)

type t = { window : Recipe_pool.t; count_include_pad : bool }

let cascade c = { c with window = Recipe_pool.cascade c.window }
let self_shape c = Recipe_pool.self_shape c.window
let kernel_size c = Recipe_pool.kernel_size c.window
let strides c = Recipe_pool.strides c.window
let pads c = Recipe_pool.pads c.window
let ceil_mode c = Recipe_pool.ceil_mode c.window
let count_include_pad c = c.count_include_pad

let axes ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h ~pad_w ~n ~c ~input_h
    ~input_w ~ceil_mode ~count_include_pad =
  let lift (a : Recipe_pool.t Walk.axis) : t Walk.axis =
    {
      Walk.name = a.name;
      mutate =
        (fun pcg cfg ->
          let w, pcg = a.mutate pcg cfg.window in
          ({ cfg with window = w }, pcg));
    }
  in
  Walk.field_axis "count_include_pad" count_include_pad (fun c v ->
      { c with count_include_pad = v })
  :: List.map lift
       (Recipe_pool.axes ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h ~pad_w
          ~n ~c ~input_h ~input_w ~ceil_mode)

let pp ppf c =
  Format.fprintf ppf "%a count_include_pad=%b" Recipe_pool.pp c.window
    c.count_include_pad
