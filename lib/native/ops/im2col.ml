(* The paired 2-D column operators.  The bridge presents images in the
   channel-last working layout [D=batch,H=height,W=width,C=channels].
   im2col's ATen rank-3 result is [H=batch,W=channels*kh*kw,C=oh*ow]; col2im
   consumes that exact layout and restores the image layout. *)

module Window = struct
  type t = {
    kernel : Dim.extent Dim.t;
    dilation : Op_config.Pos.t;
    pad : Op_config.Nonneg.t;
    stride : Op_config.Pos.t;
  }

  let jsont : t Jsont.t =
    Jsont.Object.map ~kind:"im2col_window" (fun kernel dilation pad stride ->
        { kernel; dilation; pad; stride })
    |> Jsont.Object.mem "kernel" Dim.extent_jsont ~enc:(fun x -> x.kernel)
    |> Jsont.Object.mem "dilation" Op_config.Pos.jsont ~enc:(fun x ->
        x.dilation)
    |> Jsont.Object.mem "pad" Op_config.Nonneg.jsont ~enc:(fun x -> x.pad)
    |> Jsont.Object.mem "stride" Op_config.Pos.jsont ~enc:(fun x -> x.stride)
    |> Jsont.Object.finish

  let pp ppf x =
    Fmt.pf ppf "{kernel=%a; dilation=%a; pad=%a; stride=%a}" Dim.pp x.kernel
      Op_config.Pos.pp x.dilation Op_config.Nonneg.pp x.pad Op_config.Pos.pp
      x.stride

  let output_extent ~in_extent x =
    Window_axis.output_extent ~ceil_mode:false ~kernel:x.kernel ~stride:x.stride
      ~pad_before:x.pad ~pad_after:x.pad ~dilation:x.dilation ~in_extent
end

module Params = struct
  type t = { h : Window.t; w : Window.t }

  let jsont : t Jsont.t =
    Jsont.Object.map ~kind:"im2col_params" (fun h w -> { h; w })
    |> Jsont.Object.mem "h" Window.jsont ~enc:(fun x -> x.h)
    |> Jsont.Object.mem "w" Window.jsont ~enc:(fun x -> x.w)
    |> Jsont.Object.finish

  let pp ppf x = Fmt.pf ppf "{h=%a; w=%a}" Window.pp x.h Window.pp x.w
end

let one shape axes =
  List.for_all (fun axis -> Dim.equal (Vec6.get shape axis) Dim.one) axes

let column_channels params =
  (params.Params.h.Window.kernel :> int) * (params.w.Window.kernel :> int)

module Im2col = struct
  type t = { params : Params.t; x : Tensor_ref.t }

  let name = "Im2col"

  let jsont =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" Params.jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun x ->
        Json_util.jobj
          [
            ("params", Json_util.enc Params.jsont x.params);
            ("x", Json_util.enc Tensor_ref.jsont x.x);
          ])
      Jsont.json

  let operands x = [ x.x ]
  let map_operands f x = { x with x = f x.x }

  let pp pp_ref ppf x =
    Fmt.pf ppf "@[<hv 2>im2col@ x=%a@ params=%a@]" pp_ref x.x Params.pp x.params

  let output_shape ~x_shape params =
    let open Err.Syntax in
    let* () =
      if one x_shape [ Axis.N; Axis.T ] then Err.return ()
      else Err.fail (`Im2col Shape_error.Im2col.{ fault = `Image_input_rank })
    in
    let* oh =
      Window.output_extent ~in_extent:(Vec6.get x_shape Axis.H) params.Params.h
    in
    let* ow =
      Window.output_extent ~in_extent:(Vec6.get x_shape Axis.W) params.w
    in
    let channels = (Vec6.get x_shape Axis.C :> int) * column_channels params in
    let locations = (oh :> int) * (ow :> int) in
    Err.return
      (Vec6.shape ~n:1 ~t:1 ~d:1
         ~h:(Vec6.get x_shape Axis.D :> int)
         ~w:channels ~c:locations)

  module Compute (S : Semantics.SEMANTICS) = struct
    let bounded raw (extent : Dim.extent Dim.t) =
      let hi = S.index_const ((extent :> int) - 1) in
      S.index_min (S.index_max raw (S.of_index S.index_zero)) hi

    let pixel params ~x_shape ~x out =
      let kh = (params.Params.h.Window.kernel :> int)
      and kw = (params.w.Window.kernel :> int) in
      let per = kh * kw in
      let q = S.of_index (Vec6.get out Axis.W) in
      let channel = S.index_floor_div_pos q (Op_config.Pos.of_int per) in
      let q_rem = S.index_add q (S.index_scale (-per) channel) in
      let kernel_h = S.index_floor_div_pos q_rem (Op_config.Pos.of_int kw) in
      let kernel_w = S.index_add q_rem (S.index_scale (-kw) kernel_h) in
      let ow_extent =
        match
          Window.output_extent ~in_extent:(Vec6.get x_shape Axis.W) params.w
        with
        | Ok x -> x
        | Error _ -> assert false
      in
      let loc = S.of_index (Vec6.get out Axis.C) in
      let oh =
        S.index_floor_div_pos loc (Op_config.Pos.of_int (ow_extent :> int))
      in
      let ow = S.index_add loc (S.index_scale (-(ow_extent :> int)) oh) in
      let source window output kernel =
        S.index_add
          (S.index_add
             (S.index_scale (window.Window.stride :> int) output)
             (S.index_scale (window.dilation :> int) kernel))
          (S.index_const (-((window.pad : Op_config.Nonneg.t) :> int)))
      in
      let ih = source params.h oh kernel_h
      and iw = source params.w ow kernel_w in
      let bh = bounded ih (Vec6.get x_shape Axis.H)
      and bw = bounded iw (Vec6.get x_shape Axis.W) in
      let value =
        S.load6 x ~n:S.index_zero ~t:S.index_zero ~d:(Vec6.get out Axis.H)
          ~h:(S.assume_index bh) ~w:(S.assume_index bw)
          ~c:(S.assume_index channel)
      in
      S.select (S.index_eq ih bh)
        (S.select (S.index_eq iw bw) value (S.const 0.))
        (S.const 0.)
  end
end

module Col2im = struct
  type params = {
    window : Params.t;
    output_h : Dim.extent Dim.t;
    output_w : Dim.extent Dim.t;
  }

  let params_jsont =
    Jsont.Object.map ~kind:"col2im_params" (fun window output_h output_w ->
        { window; output_h; output_w })
    |> Jsont.Object.mem "window" Params.jsont ~enc:(fun x -> x.window)
    |> Jsont.Object.mem "output_h" Dim.extent_jsont ~enc:(fun x -> x.output_h)
    |> Jsont.Object.mem "output_w" Dim.extent_jsont ~enc:(fun x -> x.output_w)
    |> Jsont.Object.finish

  let pp_params ppf x =
    Fmt.pf ppf "{window=%a; output_h=%a; output_w=%a}" Params.pp x.window Dim.pp
      x.output_h Dim.pp x.output_w

  type t = { params : params; x : Tensor_ref.t }

  let name = "Col2im"

  let jsont =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun x ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont x.params);
            ("x", Json_util.enc Tensor_ref.jsont x.x);
          ])
      Jsont.json

  let operands x = [ x.x ]
  let map_operands f x = { x with x = f x.x }

  let pp pp_ref ppf x =
    Fmt.pf ppf "@[<hv 2>col2im@ x=%a@ params=%a@]" pp_ref x.x pp_params x.params

  let output_shape ~x_shape params =
    let open Err.Syntax in
    let* () =
      if one x_shape [ Axis.N; Axis.T; Axis.D ] then Err.return ()
      else Err.fail (`Im2col Shape_error.Im2col.{ fault = `Col_input_rank })
    in
    let per_channel = column_channels params.window in
    let encoded_channels =
      ((Vec6.get x_shape Axis.W : Dim.extent Dim.t) :> int)
    in
    let* () =
      if encoded_channels mod per_channel = 0 then Err.return ()
      else Err.fail (`Im2col Shape_error.Im2col.{ fault = `Column_channels })
    in
    let* oh = Window.output_extent ~in_extent:params.output_h params.window.h in
    let* ow = Window.output_extent ~in_extent:params.output_w params.window.w in
    let* () =
      if
        Dim.equal (Vec6.get x_shape Axis.C)
          (Dim.extent ((oh :> int) * (ow :> int)))
      then Err.return ()
      else Err.fail (`Im2col Shape_error.Im2col.{ fault = `Column_locations })
    in
    Err.return
      (Vec6.shape ~n:1 ~t:1
         ~d:(Vec6.get x_shape Axis.H :> int)
         ~h:(params.output_h :> int)
         ~w:(params.output_w :> int)
         ~c:(encoded_channels / per_channel))

  module Compute (S : Semantics.SEMANTICS) = struct
    let bounded raw (extent : Dim.extent Dim.t) =
      S.index_min
        (S.index_max raw (S.of_index S.index_zero))
        (S.index_const ((extent :> int) - 1))

    let pixel params ~x_shape:_ ~x out =
      let oh =
        match
          Window.output_extent ~in_extent:params.output_h params.window.h
        with
        | Ok x -> x
        | Error _ -> assert false
      and ow =
        match
          Window.output_extent ~in_extent:params.output_w params.window.w
        with
        | Ok x -> x
        | Error _ -> assert false
      in
      let kh_extent = params.window.h.Window.kernel
      and kw_extent = params.window.w.Window.kernel in
      let per = (kh_extent :> int) * (kw_extent :> int) in
      let output_coord axis = S.of_index (Vec6.get out axis) in
      let candidate window coordinate kernel =
        let raw =
          S.index_add
            (S.index_add coordinate
               (S.index_const ((window.Window.pad : Op_config.Nonneg.t) :> int)))
            (S.index_scale
               (-((window.dilation : Op_config.Pos.t) :> int))
               kernel)
        in
        let q = S.index_floor_div_pos raw window.stride in
        (raw, q)
      in
      S.sum ~lo:S.index_zero ~hi:(S.index_extent kh_extent) (fun kh ->
          S.sum ~lo:S.index_zero ~hi:(S.index_extent kw_extent) (fun kw ->
              let raw_h, win_h =
                candidate params.window.h (output_coord Axis.H) (S.of_index kh)
              in
              let raw_w, win_w =
                candidate params.window.w (output_coord Axis.W) (S.of_index kw)
              in
              let safe_h = bounded win_h oh and safe_w = bounded win_w ow in
              let channel = output_coord Axis.C in
              let column_channel =
                S.index_add
                  (S.index_add
                     (S.index_scale per channel)
                     (S.index_scale (kw_extent :> int) (S.of_index kh)))
                  (S.of_index kw)
              in
              let location =
                S.index_add (S.index_scale (ow :> int) safe_h) safe_w
              in
              let value =
                S.load6 x ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
                  ~h:(Vec6.get out Axis.D)
                  ~w:(S.assume_index column_channel)
                  ~c:(S.assume_index location)
              in
              let exact_h =
                S.index_eq raw_h
                  (S.index_scale (params.window.h.Window.stride :> int) win_h)
              and exact_w =
                S.index_eq raw_w
                  (S.index_scale (params.window.w.Window.stride :> int) win_w)
              in
              S.select exact_h
                (S.select exact_w
                   (S.select (S.index_eq win_h safe_h)
                      (S.select (S.index_eq win_w safe_w) value (S.const 0.))
                      (S.const 0.))
                   (S.const 0.))
                (S.const 0.)))
  end
end
