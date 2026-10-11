open Err.Syntax
module J = Json

let ( >>= ) = Err.bind

type interpolation = Bicubic | Bilinear
type crop_mode = Border | Center | Squash

let round_even x =
  let lo = Float.floor x in
  let frac = x -. lo in
  if frac < 0.5 then int_of_float lo
  else if frac > 0.5 then int_of_float lo + 1
  else
    let n = int_of_float lo in
    if n mod 2 = 0 then n else n + 1

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

module Recipe = struct
  type t = {
    fill : int array;
    height : int;
    interpolation : interpolation;
    mean : float array;
    mode : crop_mode;
    scale_height : int;
    scale_width : int;
    std : float array;
    width : int;
  }

  let numbers j = J.array j >>= Err.List.map J.number

  let of_json preprocessing =
    let* j = J.member "recipe" preprocessing in
    let* () = J.version 1 j in
    let* backend = J.field "backend" j in
    let* () = J.same "resize backend" backend "PIL" in
    let* antialias = J.field "antialias" j in
    let* () = J.same "antialias" antialias "PIL_always" in
    let* operations =
      J.member "operations" j >>= J.array >>= Err.List.map J.string
    in
    let* () =
      J.same "preprocessing operations" operations
        [
          "RGB_PIL_reconstruction";
          "resize_uint8";
          "center_crop_or_pad_uint8";
          "CHW_float32";
          "divide_float32_255";
          "subtract_float32_mean";
          "divide_float32_std";
        ]
    in
    let* mode =
      J.field "crop_mode_resolved" j >>= function
      | "border" -> Ok Border
      | "center" -> Ok Center
      | "squash" -> Ok Squash
      | _ -> J.bad "unsupported crop mode"
    in
    let* interpolation =
      J.field "interpolation" j >>= function
      | "bicubic" -> Ok Bicubic
      | "bilinear" -> Ok Bilinear
      | _ -> J.bad "unsupported interpolation"
    in
    let* config = J.member "config" j in
    let* dims = J.member "input_size" config >>= J.shape in
    let* height, width =
      match dims with
      | [ 3; h; w ] when h <= 4096 && w <= 4096 && h <= 11184810 / w -> Ok (h, w)
      | _ -> J.bad "unsupported RGB target dimensions"
    in
    let* mean = J.member "mean" config >>= numbers in
    let* std = J.member "std" config >>= numbers in
    let* () =
      if
        List.length mean = 3
        && List.length std = 3
        && List.for_all (fun x -> Float.is_finite (f32 x)) mean
        && List.for_all (fun x -> f32 x > 0. && Float.is_finite (f32 x)) std
      then Ok ()
      else J.bad "invalid normalization"
    in
    let* pct = J.member "crop_pct" config in
    let* pct = match pct with Jsont.Null _ -> Ok 0.875 | _ -> J.number pct in
    let pct = if pct = 0. then 0.875 else pct in
    let* () =
      if pct <= 0. || pct > 2. then J.bad "unsupported crop percentage"
      else Ok ()
    in
    let scale_height = int_of_float (Float.floor (float_of_int height /. pct))
    and scale_width = int_of_float (Float.floor (float_of_int width /. pct)) in
    let* () =
      if min scale_height scale_width < 1 || max scale_height scale_width > 8192
      then J.bad "resize exceeds dimension limit"
      else Ok ()
    in
    let* declared = J.member "scale_size" j >>= J.shape in
    let* () =
      J.same "scale dimensions" declared [ scale_height; scale_width ]
    in
    let fill =
      Array.of_list
        (List.map
           (fun v -> if mode = Border then round_even (255. *. v) else 0)
           mean)
    in
    let* declared_fill =
      J.member "padding_fill" j >>= J.array >>= Err.List.map J.integer
    in
    let* () = J.same "padding fill" declared_fill (Array.to_list fill) in
    let* () =
      if Array.for_all (fun x -> x >= 0 && x <= 255) fill then Ok ()
      else J.bad "invalid padding pixel"
    in
    let* () =
      Err.List.iter
        (fun (field, expected) ->
          let* actual = J.field field j in
          J.same field actual expected)
        [
          ("crop_rounding", "ties_to_even");
          ("padding", "extra_pixel_right_bottom");
          ( "resize_rounding",
            if mode = Center && height = width then
              "truncate_secondary_dimension"
            else "ties_to_even" );
        ]
    in
    let+ () =
      J.member "pipeline" j >>= J.array >>= Err.List.map J.string
      >>= J.same "pipeline"
            [
              (if mode = Border || (mode = Center && scale_height <> scale_width)
               then "ResizeKeepRatio"
               else "Resize");
              (if mode = Border then "CenterCropOrPad" else "CenterCrop");
              "MaybeToTensor";
              "Normalize";
            ]
    in
    {
      fill;
      height;
      interpolation;
      mean = Array.of_list (List.map f32 mean);
      mode;
      scale_height;
      scale_width;
      std = Array.of_list (List.map f32 std);
      width;
    }
end

module Image = struct
  type t = { data : bytes; height : int; width : int }

  let of_tensor (t : Pt2_tensor.t) =
    let* () =
      if t.dtype = Pt2_dtype.UInt8 then Ok ()
      else J.bad "decoded image is not uint8"
    in
    let* h, w =
      match t.sizes with
      | [ 3; h; w ]
        when h > 0 && w > 0 && h <= 8192 && w <= 8192 && h <= 44739242 / w ->
          Ok (h, w)
      | _ -> J.bad "decoded image dimensions exceed limit"
    in
    let+ logical =
      Pt2_fixture.Logical.of_pt2 ~max_bytes:134217728 t
      |> Err.map_error (fun e ->
          `Timm_fixture (Fmt.str "%a" Pt2_fixture.Logical.pp_error e))
    in
    {
      data =
        Bytes.init (3 * h * w) (fun i -> Bigarray.Array1.get logical.data i);
      height = h;
      width = w;
    }

  let get t c y x =
    Char.code (Bytes.get t.data ((((c * t.height) + y) * t.width) + x))
end

module Coefficients = struct
  type t = { start : int; weights : int array }

  let kernel interpolation x =
    let x = Float.abs x in
    match interpolation with
    | Bicubic ->
        if x < 1. then (((1.5 *. x) -. 2.5) *. x *. x) +. 1.
        else if x < 2. then (((((x -. 5.) *. x) +. 8.) *. x) -. 4.) *. -0.5
        else 0.
    | Bilinear -> if x < 1. then 1. -. x else 0.

  (* Pillow 12.3.0 Resample.c: normalize in double, quantize to signed
     22-bit coefficients, then round/clip each uint8 separable pass. *)
  let create interpolation input output =
    let scale = float_of_int input /. float_of_int output in
    let filter_scale = max 1. scale in
    let support =
      filter_scale *. match interpolation with Bicubic -> 2. | Bilinear -> 1.
    in
    let inverse = 1. /. filter_scale in
    Array.init output (fun i ->
        let center = (float_of_int i +. 0.5) *. scale in
        let start = max 0 (int_of_float (center -. support +. 0.5)) in
        let stop = min input (int_of_float (center +. support +. 0.5)) in
        let weights =
          Array.init (stop - start) (fun x ->
              kernel interpolation
                ((float_of_int (x + start) -. center +. 0.5) *. inverse))
        in
        let sum = Array.fold_left ( +. ) 0. weights in
        let weights =
          Array.map
            (fun w ->
              let w = if sum = 0. then w else w /. sum in
              int_of_float ((w *. 4194304.) +. if w < 0. then -0.5 else 0.5))
            weights
        in
        { start; weights })
end

let resize interpolation image ~height ~width =
  let pass horizontal target image =
    let source = if horizontal then image.Image.width else image.height in
    if target = source then image
    else
      let coefficients = Coefficients.create interpolation source target in
      let height, width =
        if horizontal then (image.height, target) else (target, image.width)
      in
      let data = Bytes.create (3 * height * width) in
      for c = 0 to 2 do
        for y = 0 to height - 1 do
          for x = 0 to width - 1 do
            let coeff = coefficients.(if horizontal then x else y) in
            let sum = ref 2097152L in
            Array.iteri
              (fun k weight ->
                let v =
                  if horizontal then Image.get image c y (coeff.start + k)
                  else Image.get image c (coeff.start + k) x
                in
                sum :=
                  Int64.add !sum
                    (Int64.mul (Int64.of_int v) (Int64.of_int weight)))
              coeff.weights;
            let pixel =
              min 255 (max 0 (Int64.to_int (Int64.shift_right !sum 22)))
            in
            Bytes.set data ((((c * height) + y) * width) + x) (Char.chr pixel)
          done
        done
      done;
      { Image.data; height; width }
  in
  pass false height (pass true width image)

module Event = struct
  type t = {
    after : int list;
    before : int list;
    extra : (string * int list) list;
    fill : int array option;
    op : string;
  }

  let check e j =
    let* op = J.field "op" j in
    let* before = J.member "before" j >>= J.array >>= Err.List.map J.integer in
    let* after = J.member "after" j >>= J.array >>= Err.List.map J.integer in
    let* () = J.same "geometry operation" e.op op in
    let* () = J.same "geometry source" e.before before in
    let* () = J.same "geometry result" e.after after in
    let* () =
      Err.List.iter
        (fun (k, expected) ->
          let* actual = J.member k j >>= J.array >>= Err.List.map J.integer in
          J.same k actual expected)
        e.extra
    in
    match e.fill with
    | None -> Ok ()
    | Some expected ->
        let* j = J.member "fill" j in
        let* actual =
          match j with
          | Jsont.Array _ -> J.array j >>= Err.List.map J.integer
          | _ ->
              let+ n = J.integer j in
              [ n; n; n ]
        in
        J.same "geometry padding fill" actual (Array.to_list expected)
end

let run r tensor ~geometry =
  let* image = Image.of_tensor tensor in
  let h = image.Image.height and w = image.width in
  let rh, rw =
    match r.Recipe.mode with
    | Border ->
        let ratio =
          max
            (float_of_int h /. float_of_int r.scale_height)
            (float_of_int w /. float_of_int r.scale_width)
        in
        ( round_even (float_of_int h /. ratio),
          round_even (float_of_int w /. ratio) )
    | Center ->
        if r.scale_height = r.scale_width then
          if h <= w then
            ( r.scale_height,
              int_of_float
                (float_of_int r.scale_height *. float_of_int w /. float_of_int h)
            )
          else
            ( int_of_float
                (float_of_int r.scale_width *. float_of_int h /. float_of_int w),
              r.scale_width )
        else
          let ratio =
            min
              (float_of_int h /. float_of_int r.scale_height)
              (float_of_int w /. float_of_int r.scale_width)
          in
          ( round_even (float_of_int h /. ratio),
            round_even (float_of_int w /. ratio) )
    | Squash -> (r.scale_height, r.scale_width)
  in
  let* () =
    if min rh rw < 1 || max rh rw > 8192 || rh > 44739242 / rw then
      J.bad "resized image exceeds byte limit"
    else Ok ()
  in
  let image = resize r.interpolation image ~height:rh ~width:rw in
  let events =
    [
      {
        Event.after = [ rh; rw ];
        before = [ h; w ];
        extra = [];
        fill = None;
        op = "resize";
      };
    ]
  in
  let dh = max 0 (r.height - rh) and dw = max 0 (r.width - rw) in
  let ph = rh + dh and pw = rw + dw in
  let top_pad = dh / 2 and left_pad = dw / 2 in
  let events =
    if dh = 0 && dw = 0 then events
    else
      events
      @ [
          {
            Event.after = [ ph; pw ];
            before = [ rh; rw ];
            extra =
              [
                ("padding", [ left_pad; top_pad; dw - left_pad; dh - top_pad ]);
              ];
            fill = Some r.fill;
            op = "pad";
          };
        ]
  in
  let top = round_even (float_of_int (ph - r.height) /. 2.)
  and left = round_even (float_of_int (pw - r.width) /. 2.) in
  let events =
    if (dh > 0 || dw > 0) && ph = r.height && pw = r.width then events
    else
      events
      @ [
          {
            Event.after = [ r.height; r.width ];
            before = [ ph; pw ];
            extra = [ ("box", [ top; left; r.height; r.width ]) ];
            fill = None;
            op = "crop";
          };
        ]
  in
  let* published = J.array geometry in
  let* () =
    Err.List.iter2
      ~unequal_lengths:(fun _ _ -> `Timm_fixture "geometry event count differs")
      Event.check events published
  in
  let bytes = Bytes.create (3 * r.height * r.width * 4) in
  for c = 0 to 2 do
    for y = 0 to r.height - 1 do
      for x = 0 to r.width - 1 do
        let sy = y + top - top_pad and sx = x + left - left_pad in
        let pixel =
          if sy < 0 || sx < 0 || sy >= rh || sx >= rw then r.fill.(c)
          else Image.get image c sy sx
        in
        let scaled = f32 (float_of_int pixel /. 255.) in
        let centered = f32 (scaled -. r.mean.(c)) in
        let normalized = f32 (centered /. r.std.(c)) in
        Bytes.set_int32_le bytes
          (((((c * r.height) + y) * r.width) + x) * 4)
          (Int32.bits_of_float normalized)
      done
    done
  done;
  let shape = [ 3; r.height; r.width ] in
  Ok
    {
      Pt2_tensor.data = Pt2_storage.of_string (Bytes.to_string bytes);
      dtype = Pt2_dtype.Float32;
      sizes = shape;
      storage_offset = 0;
      strides = Pt2_tensor.contiguous_strides shape;
    }
