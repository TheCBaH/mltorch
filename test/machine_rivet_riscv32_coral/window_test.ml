(* The RVV window kernel on the core, bitwise against the nested-sum oracle:
   the Loop interpreter at binary32 over the convolution graph. The fixture of
   [test/coral_window] (2x2, stride 2, two channels, all-ones weights: nested
   sums give 1, a flat sum 0) runs first; the rest are stride-2 one-tap and
   wider non-overlapping windows over values that make the summation order
   show, with the count of outputs a flat sum would get wrong beside each. *)

open Graph_ir
module F = Native_test.Graph_fixtures
module W = Coralnpu_kernels.Rvv_window
module H = Kernel_harness

type case = {
  name : string;
  in_h : int;
  in_w : int;
  cin : int;
  kh : int;
  kw : int;
  sh : int;
  sw : int;
  cout : int;
}

let graph c =
  let axis kernel stride = F.conv_axis ~kernel ~stride ~pad:0 in
  F.build c.name
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:c.in_h ~w:c.in_w ~c:c.cin) () in
      let* w = constant ~shape:(F.s c.cout 1 1 c.kh c.kw c.cin) () in
      let* b = constant ~shape:(F.s1c c.cout) () in
      conv2d
        {
          Conv.Conv2d.h = axis c.kh c.sh;
          w = axis c.kw c.sw;
          in_channels = Dim.extent c.cin;
          groups = Op_config.Pos.of_int 1;
        }
        ~x ~weight:w ~bias:b ())

let window c =
  let e = Dim.extent in
  {
    W.Window.channels = e c.cin;
    in_h = e c.in_h;
    in_w = e c.in_w;
    kernel_h = e c.kh;
    kernel_w = e c.kw;
    out_channels = e c.cout;
    stride_h = e c.sh;
    stride_w = e c.sw;
  }

let bundle g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

(* The graph's weights, as [cout][kh][kw][cin]; its bias, [cout]; its input,
   [h][w][cin]: the flat arrays the tensors and the kernel are made from. *)
let oracle c ~x ~weights ~bias =
  let g = graph c in
  let b = bundle g in
  let sig_of id = Tensor_id.Map.find id g.Graph.tensors in
  let tensor values (sg : Tensor_sig.t) =
    Tensor.materialize sg.Tensor_sig.shape (fun co ->
        values (Vec6.offset sg.Tensor_sig.shape co :> int))
  in
  (* constants in creation order: the weights, then the bias *)
  let constants =
    List.map2
      (fun id values -> (id, tensor (Array.get values) (sig_of id)))
      b.Loop_ir.Loop_bundle.constants [ weights; bias ]
  in
  let inputs =
    List.map
      (fun id -> (id, tensor (Array.get x) (sig_of id)))
      b.Loop_ir.Loop_bundle.inputs
  in
  match
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle_oracle.pp_error
      (Loop_ir.Loop_bundle_oracle.run b
         ~constants:(fun id -> List.assoc_opt id constants)
         ~inputs:(fun id -> List.assoc_opt id inputs))
  with
  | [ y ] -> y
  | _ -> failwith "one output"

let r32 = F32_math.round32

(* Both references in binary32, products rounded one at a time: the nested sum
   the convolution is, and the flat sum over the same taps in the same order. *)
let reference ~nested c ~oh ~ow ~x ~weights ~bias =
  Array.init
    (oh * ow * c.cout)
    (fun e ->
      let oc = e mod c.cout and p = e / c.cout in
      let oy = p / ow and ox = p mod ow in
      let tap ch ky kx =
        let xi =
          (((((oy * c.sh) + ky) * c.in_w) + (ox * c.sw) + kx) * c.cin) + ch
        in
        let wi = (((((oc * c.kh) + ky) * c.kw) + kx) * c.cin) + ch in
        r32 (x.(xi) *. weights.(wi))
      in
      let sum f n =
        List.fold_left (fun a k -> r32 (a +. f k)) 0. (List.init n Fun.id)
      in
      let total =
        if nested then
          sum
            (fun ch -> sum (fun ky -> sum (fun kx -> tap ch ky kx) c.kw) c.kh)
            c.cin
        else
          List.fold_left
            (fun a (ch, ky, kx) -> r32 (a +. tap ch ky kx))
            0.
            (List.concat
               (List.init c.cin (fun ch ->
                    List.concat
                      (List.init c.kh (fun ky ->
                           List.init c.kw (fun kx -> (ch, ky, kx)))))))
      in
      r32 (total +. bias.(oc)))

let pool =
  [| 16777216.; -16777216.; 1.; -1.; 0.; -0.; 0.1; 3.7; 1e-8; -2.5; 1e8 |]

let data seed n =
  let st = Random.State.make [| seed |] in
  Array.init n (fun _ -> r32 pool.(Random.State.int st (Array.length pool)))

let same a b = Int32.bits_of_float a = Int32.bits_of_float b

let run ?(ones = false) c =
  let w = window c in
  let oh, ow =
    match W.output_extents w with
    | Ok (a, b) -> ((a :> int), (b :> int))
    | Error e -> failwith (Coralnpu_kernels.Kernel_refusal.to_string e)
  in
  let x, weights, bias =
    if ones then
      ( Array.init
          (c.in_h * c.in_w * c.cin)
          (fun i ->
            [| 1.; 16777216.; 0.; -16777216.; 0.; 0.; 0.; 0. |].(i mod 8)),
        Array.make (c.cout * c.kh * c.kw * c.cin) 1.,
        Array.make c.cout 0. )
    else
      ( data 1 (c.in_h * c.in_w * c.cin),
        data 2 (c.cout * c.kh * c.kw * c.cin),
        data 3 c.cout )
  in
  let y = oracle c ~x ~weights ~bias in
  let expected =
    Array.init
      (oh * ow * c.cout)
      (fun e ->
        let oc = e mod c.cout and p = e / c.cout in
        Tensor.read_at_raw32 y (function
          | Axis.H -> p / ow
          | W -> p mod ow
          | C -> oc
          | N | T | D -> 0))
  in
  let nested = reference ~nested:true c ~oh ~ow ~x ~weights ~bias in
  let flat = reference ~nested:false c ~oh ~ow ~x ~weights ~bias in
  let differ a b =
    let n = ref 0 in
    Array.iteri (fun i v -> if not (same v b.(i)) then incr n) a;
    !n
  in
  (* the kernel's weights: channel, row, column, then output channel *)
  let packed =
    Array.init
      (c.cin * c.kh * c.kw * c.cout)
      (fun i ->
        let oc = i mod c.cout in
        let t = i / c.cout in
        let ch = t / (c.kh * c.kw) in
        let ky = t / c.kw mod c.kh and kx = t mod c.kw in
        weights.((((((oc * c.kh) + ky) * c.kw) + kx) * c.cin) + ch))
  in
  let status =
    match Coralnpu_kernels.Rvv_window.module_ w with
    | Error e -> "refused: " ^ Coralnpu_kernels.Kernel_refusal.to_string e
    | Ok m -> (
        match
          H.run ~pad:64 ~symbol:(W.symbol w) ~modules:[ m ]
            ~inputs:[ H.f32_bytes x; H.f32_bytes packed; H.f32_bytes bias ]
            ~outputs:[ 4 * Array.length expected ]
            ()
        with
        | Error e -> e
        | Ok { outputs = out :: _; _ } ->
            let got = H.f32_array out in
            Fmt.str "kernel vs oracle: %d of %d differ" (differ got expected)
              (Array.length expected)
        | Ok _ -> "no output")
  in
  Fmt.pr
    "%s: %s; own nested reference vs oracle: %d differ; flat sum vs oracle: %d \
     differ@."
    c.name status (differ nested expected) (differ flat expected);
  if ones then Fmt.pr "  output %g@." expected.(0)

let case name ~in_hw ~cin ~k ~s ~cout =
  {
    name;
    in_h = fst in_hw;
    in_w = snd in_hw;
    cin;
    kh = fst k;
    kw = snd k;
    sh = fst s;
    sw = snd s;
    cout;
  }

let%expect_test "the window kernel against the nested-sum oracle" =
  run ~ones:true
    (case "counterexample" ~in_hw:(2, 2) ~cin:2 ~k:(2, 2) ~s:(2, 2) ~cout:1);
  List.iter run
    [
      case "stride 2, one tap" ~in_hw:(5, 6) ~cin:3 ~k:(1, 1) ~s:(2, 2) ~cout:4;
      case "2x2, stride 2" ~in_hw:(6, 6) ~cin:5 ~k:(2, 2) ~s:(2, 2) ~cout:6;
      case "3x3, stride 3" ~in_hw:(6, 9) ~cin:4 ~k:(3, 3) ~s:(3, 3) ~cout:16;
      case "2x3 inside 3x4" ~in_hw:(7, 9) ~cin:3 ~k:(2, 3) ~s:(3, 4) ~cout:3;
    ];
  [%expect
    {|
    counterexample: kernel vs oracle: 0 of 1 differ; own nested reference vs oracle: 0 differ; flat sum vs oracle: 1 differ
      output 1
    stride 2, one tap: kernel vs oracle: 0 of 36 differ; own nested reference vs oracle: 0 differ; flat sum vs oracle: 0 differ
    2x2, stride 2: kernel vs oracle: 0 of 54 differ; own nested reference vs oracle: 0 differ; flat sum vs oracle: 15 differ
    3x3, stride 3: kernel vs oracle: 0 of 96 differ; own nested reference vs oracle: 0 differ; flat sum vs oracle: 32 differ
    2x3 inside 3x4: kernel vs oracle: 0 of 12 differ; own nested reference vs oracle: 0 differ; flat sum vs oracle: 4 differ |}]
