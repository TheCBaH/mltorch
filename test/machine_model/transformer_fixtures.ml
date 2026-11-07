module F = Native_test.Graph_fixtures

(* Transformer elementwise and normalization graphs: residual add, RMS norm,
   and tanh-approximated GELU gated by an up projection. The 2x160, 1x67 and
   1x33 shapes exercise full vectors and tails through each target's regular
   lowering. *)
let fixture ?fmt name build =
  List.map
    (fun (rows, hidden) ->
      (Fmt.str "%s %dx%d" name rows hidden, fun () -> build ?fmt ~rows ~hidden))
    [ (2, 160); (1, 67); (1, 33) ]

let residual_add ?fmt ~rows ~hidden () =
  F.build "residual_add"
    Graph_builder.(
      let* a = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* b = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      add a b)

let rms_norm_graph ?fmt ~rows ~hidden () =
  F.build "rms_norm"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* w = constant ~shape:(F.s 1 1 1 1 1 hidden) ?fmt () in
      rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-6 } ~x ~weight:w ())

let tanh_gelu_mul ?fmt ~rows ~hidden () =
  F.build "tanh_gelu_mul"
    Graph_builder.(
      let* gate = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* up = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* g = gelu Pointwise.Gelu.Tanh gate in
      mul g up)

let variants ?fmt suffix =
  let apply f ?fmt ~rows ~hidden = f ?fmt ~rows ~hidden () in
  fixture ?fmt ("residual add" ^ suffix) (apply residual_add)
  @ fixture ?fmt ("rms norm" ^ suffix) (apply rms_norm_graph)
  @ fixture ?fmt ("tanh gelu mul" ^ suffix) (apply tanh_gelu_mul)

let f32 = variants ""

(* The same three with bfloat16 inputs and float32 outputs. *)
let bf16 = variants ~fmt:(Payload.Fmt Payload.BF16) " bf16"
