module F = Native_test.Graph_fixtures

(* Transformer elementwise and normalization graphs: residual add, RMS norm,
   and tanh-approximated GELU gated by an up projection. The 2x160, 1x67 and
   1x33 shapes exercise full vectors and tails through each target's regular
   lowering. *)
let fixture ?fmt ~narrow name build =
  List.map
    (fun (rows, hidden) ->
      ( Fmt.str "%s %dx%d" name rows hidden,
        fun () -> build ?fmt ~narrow ~rows ~hidden ))
    [ (2, 160); (1, 67); (1, 33) ]

(* The result, or, when [narrow], the result stored as bfloat16: the released
   Gemma kernels' storage contract. *)
let result ~narrow r =
  if narrow then Graph_builder.to_copy Pointwise.To_copy.Bfloat16 r
  else Graph_builder.return r

let residual_add ?fmt ~narrow ~rows ~hidden () =
  F.build "residual_add"
    Graph_builder.(
      let* a = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* b = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* r = add a b in
      result ~narrow r)

let rms_norm_graph ?fmt ~narrow ~rows ~hidden () =
  F.build "rms_norm"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* w = constant ~shape:(F.s 1 1 1 1 1 hidden) ?fmt () in
      let* r =
        rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-6 } ~x ~weight:w ()
      in
      result ~narrow r)

let tanh_gelu_mul ?fmt ~narrow ~rows ~hidden () =
  F.build "tanh_gelu_mul"
    Graph_builder.(
      let* gate = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* up = input ~shape:(F.s 1 1 1 1 rows hidden) ?fmt () in
      let* g = gelu Pointwise.Gelu.Tanh gate in
      let* r = mul g up in
      result ~narrow r)

let variants ?fmt ?(narrow = false) suffix =
  let apply f ?fmt ~narrow ~rows ~hidden = f ?fmt ~narrow ~rows ~hidden () in
  fixture ?fmt ~narrow ("residual add" ^ suffix) (apply residual_add)
  @ fixture ?fmt ~narrow ("rms norm" ^ suffix) (apply rms_norm_graph)
  @ fixture ?fmt ~narrow ("tanh gelu mul" ^ suffix) (apply tanh_gelu_mul)

let f32 = variants ""

(* The same three with bfloat16 inputs and float32 outputs. *)
let bf16 = variants ~fmt:(Payload.Fmt Payload.BF16) " bf16"

(* bfloat16 inputs and bfloat16 outputs. *)
let bf16_out =
  variants ~fmt:(Payload.Fmt Payload.BF16) ~narrow:true " bf16->bf16"
