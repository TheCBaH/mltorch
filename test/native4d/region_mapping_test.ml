open Native4d

let shape = Shape4.of_ints ~n:2 ~h:2 ~w:1 ~c:3 |> Shape4.to_vec6
let affine = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let signature id shape =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
    ~fmt:(Payload.Fmt Payload.F32) ()

let x = signature 0 shape
let weight = signature 1 affine
let bias = signature 2 affine

let operands =
  [ x; weight; bias ] |> List.to_seq
  |> Seq.map (fun (s : Tensor_sig.t) -> (s.id, s))
  |> Tensor_id.Map.of_seq

let same op =
  let operand id = Tensor_id.Map.find_opt id operands in
  let build f =
    f ~limits:Kernel.Limits.default ~output:0 ~output_shape:shape ~operand
      ~fill:(fun _ _ _ -> assert false)
    |> Err.or_raise ~pp_error:Region_computation.pp_error
  in
  let mapped = build (Region_computation4.program ~op) in
  let native =
    build
      (Region_computation.program
         ~op:(Option.get (Region_computation4.native_op op)))
  in
  Format.asprintf "%a" Region_program.pp mapped
  = Format.asprintf "%a" Region_program.pp native

let%expect_test "Softmax4 maps to Native Region structure" =
  List.iter
    (fun (name, axis) ->
      let op = Op.Softmax4 { Ops4.Softmax4.params = { axis }; x = x.id } in
      Format.printf "%s same_structure=%b@." name (same op))
    [ ("n", Axis4.N); ("h", Axis4.H); ("w", Axis4.W); ("c", Axis4.C) ];
  [%expect
    {|
    n same_structure=true
    h same_structure=true
    w same_structure=true
    c same_structure=true |}]

let%expect_test "normalization maps to Native Region structure" =
  let rms =
    Op.Rms_norm
      {
        Ops4.Rms_norm.params = { dims = [ Axis4.C ]; eps = 1e-5 };
        x = x.id;
        weight = Some weight.id;
      }
  and layer =
    Op.Layer_norm
      {
        Ops4.Layer_norm.params =
          { dims = [ Axis4.H; Axis4.W; Axis4.C ]; eps = 1e-5 };
        x = x.id;
        weight = Some weight.id;
        bias = Some bias.id;
      }
  in
  Format.printf "rms=%b layer=%b@." (same rms) (same layer);
  [%expect {| rms=true layer=true |}]
