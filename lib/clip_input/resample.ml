let precision_bits = 22
let support = 2.

(* Pillow's bicubic filter, a = -0.5. *)
let filter x =
  let a = -0.5 in
  let x = Float.abs x in
  if x < 1. then ((((a +. 2.) *. x) -. (a +. 3.)) *. x *. x) +. 1.
  else if x < 2. then (((((x -. 5.) *. x) +. 8.) *. x) -. 4.) *. a
  else 0.

(* [(bounds, coefficients)] per output position: the first input index, and the
   fixed-point weights over the following inputs. *)
let coefficients ~in_size ~out_size =
  let scale = float_of_int in_size /. float_of_int out_size in
  let filterscale = Float.max scale 1. in
  let supp = support *. filterscale in
  let ksize = (int_of_float (Float.ceil supp) * 2) + 1 in
  Array.init out_size (fun xx ->
      let center = (float_of_int xx +. 0.5) *. scale in
      let ss = 1. /. filterscale in
      let xmin = max 0 (int_of_float (center -. supp +. 0.5)) in
      let xmax = min in_size (int_of_float (center +. supp +. 0.5)) - xmin in
      let w =
        Array.init ksize (fun x ->
            if x < xmax then
              filter ((float_of_int (x + xmin) -. center +. 0.5) *. ss)
            else 0.)
      in
      let total = Array.fold_left ( +. ) 0. w in
      let w = if total <> 0. then Array.map (fun v -> v /. total) w else w in
      let fixed =
        Array.map
          (fun v ->
            let scaled = v *. float_of_int (1 lsl precision_bits) in
            if v < 0. then int_of_float (-0.5 +. scaled)
            else int_of_float (0.5 +. scaled))
          w
      in
      (xmin, xmax, fixed))

let clip8 v =
  let r = v asr precision_bits in
  if r < 0 then 0 else if r > 255 then 255 else r

let horizontal (src : Ppm.t) ~width =
  let coeffs = coefficients ~in_size:src.width ~out_size:width in
  let out = Bytes.create (3 * width * src.height) in
  for y = 0 to src.height - 1 do
    for x = 0 to width - 1 do
      let xmin, xmax, k = coeffs.(x) in
      for c = 0 to 2 do
        let acc = ref (1 lsl (precision_bits - 1)) in
        for i = 0 to xmax - 1 do
          acc :=
            !acc
            + Char.code
                (Bytes.get src.rgb ((((y * src.width) + xmin + i) * 3) + c))
              * k.(i)
        done;
        Bytes.set out ((((y * width) + x) * 3) + c) (Char.chr (clip8 !acc))
      done
    done
  done;
  { Ppm.width; height = src.height; rgb = out }

let vertical (src : Ppm.t) ~height =
  let coeffs = coefficients ~in_size:src.height ~out_size:height in
  let out = Bytes.create (3 * src.width * height) in
  for y = 0 to height - 1 do
    let ymin, ymax, k = coeffs.(y) in
    for x = 0 to src.width - 1 do
      for c = 0 to 2 do
        let acc = ref (1 lsl (precision_bits - 1)) in
        for i = 0 to ymax - 1 do
          acc :=
            !acc
            + Char.code
                (Bytes.get src.rgb (((((ymin + i) * src.width) + x) * 3) + c))
              * k.(i)
        done;
        Bytes.set out ((((y * src.width) + x) * 3) + c) (Char.chr (clip8 !acc))
      done
    done
  done;
  { Ppm.width = src.width; height; rgb = out }

let bicubic (img : Ppm.t) ~width ~height =
  let img = if width <> img.width then horizontal img ~width else img in
  if height <> img.height then vertical img ~height else img
