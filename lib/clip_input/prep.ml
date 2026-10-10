let resized_size ~width ~height ~size =
  let short, long =
    if width <= height then (width, height) else (height, width)
  in
  let new_long =
    int_of_float (float_of_int size *. float_of_int long /. float_of_int short)
  in
  if width <= height then (size, new_long) else (new_long, size)

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let pixel_values (img : Ppm.t) ~size ~mean ~std =
  let w, h = resized_size ~width:img.width ~height:img.height ~size in
  let img = Resample.bicubic img ~width:w ~height:h in
  let top = (h - size) / 2 and left = (w - size) / 2 in
  Array.init
    (3 * size * size)
    (fun i ->
      let c = i / (size * size) in
      let y = i / size mod size and x = i mod size in
      let v =
        Char.code (Bytes.get img.rgb (((((top + y) * w) + left + x) * 3) + c))
      in
      round32 ((round32 (float_of_int v /. 255.) -. mean.(c)) /. std.(c)))
