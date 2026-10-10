let resized_size ~width ~height ~size =
  let short, long =
    if width <= height then (width, height) else (height, width)
  in
  let new_long =
    int_of_float (float_of_int size *. float_of_int long /. float_of_int short)
  in
  if width <= height then (size, new_long) else (new_long, size)

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let tensor (img : Ppm.t) ~filter ~resize ~crop ~flip ~mean ~std =
  let w, h = resized_size ~width:img.width ~height:img.height ~size:resize in
  let img = Resample.resize filter img ~width:w ~height:h in
  let top = (h - crop) / 2 and left = (w - crop) / 2 in
  Array.init
    (3 * crop * crop)
    (fun i ->
      let c = i / (crop * crop) in
      let y = i / crop mod crop and x = i mod crop in
      let src_c = if flip then 2 - c else c in
      let v =
        Char.code
          (Bytes.get img.rgb (((((top + y) * w) + left + x) * 3) + src_c))
      in
      round32
        (round32 (round32 (float_of_int v /. 255.) -. round32 mean.(c))
        /. round32 std.(c)))

let pixel_values img ~size ~mean ~std =
  tensor img ~filter:Resample.Bicubic ~resize:size ~crop:size ~flip:false ~mean
    ~std
