(* Project step 19 / Section A: pins the canonical batch-coordinate contract
   `_ai_/shared_multi_output_impl.md` §3.2 derives from
   `Lstm.Lstm.Computation.program`'s own [partition]/[batch] construction
   (lib/native/ops/lstm.ml:835-840), before any Region_group code exists.
   For each of the three output ordinals and both [batch_first] values,
   confirms which axis the emitter's own partition marks [Singleton] (the
   per-key "batch" axis a group's canonical key must map onto) versus
   [Whole] (varies within one key), with batch > 1 so a degenerate
   batch-one shape cannot hide a swapped axis. *)

let sig_ id shape =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
    ~fmt:(Payload.Fmt Payload.F32) ()

let show_partition ~label ~batch_first ~output =
  let batch = 3 and seq = 2 and input_size = 2 and hidden_size = 2 in
  let params : Lstm.Lstm.params = { hidden_size; input_size; batch_first } in
  let seq_shape =
    if batch_first then Vec6.shape ~n:1 ~t:1 ~d:1 ~h:batch ~w:seq ~c:input_size
    else Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:input_size
  in
  let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:hidden_size in
  let wih_shape =
    Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:input_size
  in
  let whh_shape =
    Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:hidden_size
  in
  let bias_shape = Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let input = sig_ 0 seq_shape in
  let weight_ih = sig_ 1 wih_shape in
  let weight_hh = sig_ 2 whh_shape in
  let bias_ih = sig_ 3 bias_shape in
  let bias_hh = sig_ 4 bias_shape in
  let h0 = sig_ 5 state_shape in
  let c0 = sig_ 6 state_shape in
  let direction : Lstm.Lstm.Direction_operands.t =
    { weight_ih; weight_hh; bias = Some (bias_ih, bias_hh) }
  in
  let layer : Lstm.Lstm.Layer_operands.t =
    { forward = direction; reverse = None }
  in
  let direction_shapes : Lstm.Lstm.Direction_shapes.t =
    {
      weight_ih = wih_shape;
      weight_hh = whh_shape;
      bias = Some (bias_shape, bias_shape);
    }
  in
  let out_shape, hn_shape, cn_shape =
    Err.or_raise ~pp_error:Shape_error.pp
      (Lstm.Lstm.output_shape params ~input_shape:seq_shape
         ~layers:
           [
             {
               Lstm.Lstm.Layer_shapes.forward = direction_shapes;
               reverse = None;
             };
           ]
         ~h0_shape:state_shape ~c0_shape:state_shape)
  in
  match
    Lstm.Lstm.Computation.program ~limits:Kernel.Limits.default params ~output
      ~layers:[ layer ] ~input ~h0 ~c0 ~out_shape ~hn_shape ~cn_shape
  with
  | Error e ->
      Fmt.pr "%s: rejected: %a@." label Region_group.pp_error (Err.Error.kind e)
  | Ok program ->
      let partition = Region_program.partition program in
      Fmt.pr "%s: %a@." label Region_partition.pp partition

let%expect_test
    "lstm canonical batch-coordinate contract: both layouts, all three \
     ordinals, batch=3" =
  List.iter
    (fun batch_first ->
      List.iter
        (fun output ->
          let label = Fmt.str "batch_first=%b output=%d" batch_first output in
          show_partition ~label ~batch_first ~output)
        [ 0; 1; 2 ])
    [ false; true ];
  [%expect
    {|
    batch_first=false output=0: N=singleton T=singleton D=singleton H=whole W=singleton C=whole
    batch_first=false output=1: N=singleton T=singleton D=singleton H=whole W=singleton C=whole
    batch_first=false output=2: N=singleton T=singleton D=singleton H=whole W=singleton C=whole
    batch_first=true output=0: N=singleton T=singleton D=singleton H=singleton W=whole C=whole
    batch_first=true output=1: N=singleton T=singleton D=singleton H=whole W=singleton C=whole
    batch_first=true output=2: N=singleton T=singleton D=singleton H=whole W=singleton C=whole |}]
