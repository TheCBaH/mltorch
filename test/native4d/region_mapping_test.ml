open Native4d

let shape = Shape4.of_ints ~n:2 ~h:2 ~w:1 ~c:3 |> Shape4.to_vec6
let affine = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let signature id shape =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
    ~fmt:(Payload.Fmt Payload.F32) ()

let x = signature 0 shape
let weight = signature 1 affine
let bias = signature 2 affine

(* [Sdpa]'s query is [x] itself: same N/H (so key/value can share them, per
   [Attention.Sdpa.output_shape]'s check), key's own W (2) distinct from
   query's (1) so a fixture confusing [Wq]/[Wk] would build the wrong score
   shape. [sdpa_mask] broadcasts on every axis but C (1 on N/T/D/H/W, matching
   the score's own C = key's W = 2) -- present, so [same]'s [fill] stub is
   never called. *)
let sdpa_key = signature 3 (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3)
let sdpa_value = signature 4 (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3)
let sdpa_mask = signature 5 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2)

let operands =
  [ x; weight; bias; sdpa_key; sdpa_value; sdpa_mask ]
  |> List.to_seq
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
  let same_local (l1 : Region_local.t) (l2 : Region_local.t) =
    Expr.Local_var.equal l1.Region_local.id l2.Region_local.id
    && Expr.Value.equal
         (Region_local.Rhs.value l1.Region_local.rhs)
         (Region_local.Rhs.value l2.Region_local.rhs)
  in
  let mapped_locals = Region_program.locals mapped
  and native_locals = Region_program.locals native in
  List.length mapped_locals = List.length native_locals
  && List.for_all2 same_local mapped_locals native_locals
  && Expr.Value.equal
       (Region_program.output mapped)
       (Region_program.output native)
  && Region_program.partition mapped = Region_program.partition native

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

let%expect_test "Sdpa maps to Native Region structure" =
  let sdpa =
    Op.Sdpa
      {
        Attention.Sdpa.params = { scale = Attention.Sdpa.Scale.Default };
        query = x.id;
        key = sdpa_key.id;
        value = sdpa_value.id;
        mask = Some sdpa_mask.id;
      }
  in
  Format.printf "sdpa=%b@." (same sdpa);
  [%expect {| sdpa=true |}]
