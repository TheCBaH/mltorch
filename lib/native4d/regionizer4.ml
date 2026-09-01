(* Native4D's operation boundary for Region-authored computation.  It owns only
   the dialect mapping: once a four-axis operation has been translated through
   [Graph_shape4], Native's operation-owned builder remains the sole numeric
   definition. *)

type error = Regionizer.error
type synthetic_role = Regionizer.synthetic_role

let is_region_authored : Op.t -> bool = function
  | Op.Layer_norm _ | Op.Rms_norm _ | Op.Softmax4 _ -> true
  | _ -> false

let native_op : Op.t -> Graph_ir.op option = function
  | Op.Rms_norm { Ops4.Rms_norm.params; x; weight } ->
      Some
        (Graph_ir.Rms_norm
           { Norm.RmsNorm.params = Graph_shape4.rms_params params; x; weight })
  | Op.Layer_norm { Ops4.Layer_norm.params; x; weight; bias } ->
      Some
        (Graph_ir.Layer_norm
           {
             Norm.LayerNorm.params = Graph_shape4.layer_norm_params params;
             x;
             weight;
             bias;
           })
  | Op.Softmax4 { Ops4.Softmax4.params; x } ->
      Some
        (Graph_ir.Softmax
           { Reduce.Softmax.params = Graph_shape4.softmax_params params; x })
  | _ -> None

let program ~limits ~op ~output ~output_shape ~operand ~fill =
  match native_op op with
  | Some op ->
      Regionizer.program ~limits ~op ~output ~output_shape ~operand ~fill
  | None -> assert false
