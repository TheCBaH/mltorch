type t = {
  k : int;
  s : int;
  p : int;
  d : int;
  g : int;
  inc : int;
  outc : int;
  n : int;
  len : int;
}

let cascade c =
  let inc = Walk_core.Window_math.round_up_multiple ~n:c.inc ~m:c.g
  and outc = Walk_core.Window_math.round_up_multiple ~n:c.outc ~m:c.g in
  let len =
    Walk_core.Window_math.grow_input ~in_size:c.len ~pad:c.p ~kernel:c.k
      ~dilation:c.d
  in
  { c with inc; outc; len }

let input_shape c = [ c.n; c.inc; c.len ]
let weight_shape c = [ c.outc; c.inc / c.g; c.k ]
let bias_shape c = [ c.outc ]

let axes ~k ~s ~p ~d ~g ~inc ~outc ~n ~len =
  Walk.
    [
      field_axis "k" k (fun (c : t) v -> { c with k = v });
      field_axis "s" s (fun (c : t) v -> { c with s = v });
      field_axis "p" p (fun (c : t) v -> { c with p = v });
      field_axis "d" d (fun (c : t) v -> { c with d = v });
      field_axis "g" g (fun (c : t) v -> { c with g = v });
      field_axis "inc" inc (fun (c : t) v -> { c with inc = v });
      field_axis "outc" outc (fun (c : t) v -> { c with outc = v });
      field_axis "n" n (fun (c : t) v -> { c with n = v });
      field_axis "len" len (fun (c : t) v -> { c with len = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{k=%d s=%d p=%d d=%d g=%d in=%d out=%d n=%d len=%d}" c.k
    c.s c.p c.d c.g c.inc c.outc c.n c.len
