(* Native4D's operation boundary for authored Region computation.  It owns the
   checked Axis4 mapping only; Native's operation-owned builder is the sole
   numeric definition. *)

type error = Region_computation.error
type synthetic_role = Region_computation.synthetic_role

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
  | Op.Sdpa t -> Some (Graph_ir.Sdpa t)
  | Op.Lstm t -> Some (Graph_ir.Lstm t)
  | Op.Softmax4 { Ops4.Softmax4.params; x } ->
      Some
        (Graph_ir.Softmax
           { Reduce.Softmax.params = Graph_shape4.softmax_params params; x })
  | _ -> None

(* One list to maintain, not two that can drift: an op routes to Region
   computation exactly when it maps onto a native one. *)
let is_region_authored op = native_op op <> None

let program ~limits ~op ~output ~output_shape ~operand ~fill =
  match native_op op with
  | Some op ->
      Region_computation.program ~limits ~op ~output ~output_shape ~operand
        ~fill
  | None -> assert false
