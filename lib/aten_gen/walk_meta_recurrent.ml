(* Split out of walk_meta.ml see walk_meta_entry.ml for the [t] record these
   build. *)

open Walk_meta_entry

(* lstm.input: two Tensor[] arguments (hx, params) whose lengths are
   themselves a function of layers/directions/has_biases, on top of the usual
   multi-tensor correlation -- straight into [needs_meta], since
   [default_tensor] refuses any op with a tensor-list argument at all. Every
   shape [Recipe_lstm] builds is derived from one record, so [cascade] has
   nothing to repair (Recipe_sdpa/Recipe_norm's own discipline).

   [dropout] is walked across 0/0.5/1 even though [train=false] makes it
   numerically inactive (lstm-plan.md §2): the values pin that the ATen
   fallback really does ignore it outside training, the same non-argument
   [layer_norm]'s [cudnn_enable] axis pins. *)
let lstm =
  {
    module_name = "Lstm_walk";
    target = "torch.ops.aten.lstm.input";
    recipe = "Recipe_lstm";
    initial =
      "Aten_walk_recipes.Recipe_lstm.{ batch = 3; seq = 4; input_size = 5; \
       hidden_size = 6; layers = 1; bidirectional = false; has_biases = true; \
       batch_first = false; dropout = 0.0 }";
    axes =
      "Aten_walk_recipes.Recipe_lstm.axes ~batch:[ 1; 3; 5 ] ~seq:[ 1; 4; 7 ] \
       ~input_size:[ 1; 5; 8 ] ~hidden_size:[ 1; 6; 9 ] ~layers:[ 1; 2; 3 ] \
       ~bidirectional:[ false; true ] ~has_biases:[ true; false ] \
       ~batch_first:[ false; true ] ~dropout:[ 0.0; 0.5; 1.0 ] ()";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_lstm.input_shape c) in
    let h0, pcg = Walk.tensor_spec pcg (Recipe_lstm.state_shape c) in
    let c0, pcg = Walk.tensor_spec pcg (Recipe_lstm.state_shape c) in
    let params, pcg =
      List.fold_left
        (fun (acc, pcg) shape ->
          let t, pcg = Walk.tensor_spec pcg shape in
          (t :: acc, pcg))
        ([], pcg) (Recipe_lstm.param_shapes c)
    in
    let params = List.rev params in
    ( Aten_op_spec.Op_lstm_input.(
        spec
          {
            input;
            hx = [ h0; c0 ];
            params;
            has_biases = Recipe_lstm.has_biases c;
            num_layers = Recipe_lstm.num_layers c;
            dropout = Aten_spec.Float32.to_f32 (Recipe_lstm.dropout c);
            train = false;
            bidirectional = Recipe_lstm.bidirectional c;
            batch_first = Recipe_lstm.batch_first c;
          }),
      pcg )|};
  }
