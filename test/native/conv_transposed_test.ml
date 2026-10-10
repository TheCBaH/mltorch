(* Transposed convolution against its scatter definition, written independently
   of the gather the engine computes: every input element adds
   [x * w] at [in * stride - pad + k * dilation] on each axis. The configs cover
   padding, dilation, output padding, groups and strides larger than the
   kernel (taps the output never receives). *)

open Compute_fixtures

let pos = Op_config.Pos.of_int
let nonneg = Op_config.Nonneg.of_int
let value n = float_of_int ((n * 7 mod 11) - 5)

type config = {
  stride : int * int;
  pad : int * int;
  dilation : int * int;
  output_padding : int * int;
  groups : int;
}

let kh = 3
let kw = 2
let h = 3
let w = 4
let in_channels = 4
let out_per_group = 2

let check
    { stride = sh, sw; pad = ph, pw; dilation = dh, dw; output_padding; groups }
    =
  let oph, opw = output_padding in
  let p : Conv.Convolution.params =
    {
      stride = { h = pos sh; w = pos sw };
      padding = { h = nonneg ph; w = nonneg pw };
      dilation = { h = pos dh; w = pos dw };
      transposed = true;
      output_padding = { h = nonneg oph; w = nonneg opw };
      groups = pos groups;
    }
  in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:in_channels in
  let weight_shape =
    Vec6.shape ~n:in_channels ~t:1 ~d:1 ~h:kh ~w:kw ~c:out_per_group
  in
  let out_channels = out_per_group * groups in
  let x =
    Tensor.materialize x_shape (fun c ->
        value ((((row c * w) + col c) * in_channels) + chan c))
  in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        value
          (100
          + ((((Dim.to_int (Vec6.get c Axis.N) * kh) + row c) * kw) + col c)
            * out_per_group
          + chan c))
  in
  let bias = Tensor.materialize (s1c out_channels) (fun c -> value (chan c)) in
  let module Cv = Conv.Convolution.Compute (Direct) in
  match
    Err.payload
      (eval_tensor
         (Conv.Convolution.output_shape ~x_shape ~weight_shape p)
         (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias))
  with
  | Error e -> Format.asprintf "error: %a" Shape_error.pp e
  | Ok out ->
      let out_h = ((h - 1) * sh) - (2 * ph) + (dh * (kh - 1)) + oph + 1
      and out_w = ((w - 1) * sw) - (2 * pw) + (dw * (kw - 1)) + opw + 1 in
      let expected = Array.make (out_h * out_w * out_channels) 0. in
      let at oh ow oc = (((oh * out_w) + ow) * out_channels) + oc in
      let in_per_group = in_channels / groups in
      for ih = 0 to h - 1 do
        for iw = 0 to w - 1 do
          for ic = 0 to in_channels - 1 do
            for ky = 0 to kh - 1 do
              for kx = 0 to kw - 1 do
                for lo = 0 to out_per_group - 1 do
                  let oh = (ih * sh) - ph + (ky * dh)
                  and ow = (iw * sw) - pw + (kx * dw)
                  and oc = (ic / in_per_group * out_per_group) + lo in
                  if oh >= 0 && oh < out_h && ow >= 0 && ow < out_w then
                    expected.(at oh ow oc) <-
                      expected.(at oh ow oc)
                      +. Tensor.read x
                           (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:ih ~w:iw ~c:ic)
                         *. Tensor.read weight
                              (Vec6.coord ~n:ic ~t:0 ~d:0 ~h:ky ~w:kx ~c:lo)
                done
              done
            done
          done
        done
      done;
      let bad = ref 0 in
      for oh = 0 to out_h - 1 do
        for ow = 0 to out_w - 1 do
          for oc = 0 to out_channels - 1 do
            let want =
              expected.(at oh ow oc)
              +. Tensor.read bias (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:oc)
            and got =
              Tensor.read out (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:oh ~w:ow ~c:oc)
            in
            if want <> got then incr bad
          done
        done
      done;
      Format.asprintf "%dx%d out, %d of %d differ" out_h out_w !bad
        (out_h * out_w * out_channels)

let%expect_test "transposed convolution equals its scatter definition" =
  let base =
    {
      stride = (1, 1);
      pad = (0, 0);
      dilation = (1, 1);
      output_padding = (0, 0);
      groups = 1;
    }
  in
  List.iter
    (fun (name, cfg) -> Printf.printf "%s: %s\n" name (check cfg))
    [
      ("unit stride", base);
      ("stride 2", { base with stride = (2, 2) });
      ("stride 4 over kernel 3x2", { base with stride = (4, 3) });
      ("padding", { base with pad = (1, 1) });
      ("padding and stride", { base with stride = (2, 2); pad = (1, 0) });
      ("dilation", { base with dilation = (2, 3) });
      ("output padding", { base with stride = (2, 3); output_padding = (1, 2) });
      ("two groups", { base with groups = 2 });
      ( "all of it",
        {
          stride = (2, 3);
          pad = (1, 1);
          dilation = (2, 1);
          output_padding = (1, 1);
          groups = 2;
        } );
    ];
  [%expect {|
    unit stride: 5x5 out, 0 of 50 differ
    stride 2: 7x8 out, 0 of 112 differ
    stride 4 over kernel 3x2: 11x11 out, 0 of 242 differ
    padding: 3x3 out, 0 of 18 differ
    padding and stride: 5x8 out, 0 of 80 differ
    dilation: 7x7 out, 0 of 98 differ
    output padding: 8x13 out, 0 of 208 differ
    two groups: 5x5 out, 0 of 100 differ
    all of it: 8x10 out, 0 of 320 differ |}]
