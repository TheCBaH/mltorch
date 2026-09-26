open Loop_ir
(* Hand-built programs and kernels shared by the Loop IR tests. Everything here
   is deliberately tiny: a test that needs an op-sized graph belongs to the op
   sweep, not to a foundation test. *)

let shape_w n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int
let sg id shape fmt = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ()

let buffer id shape fmt role =
  { Loop_buffer.id = tid id; sg = sg id shape fmt; role }

(* A coordinate that varies along W only, every other axis pinned at 0. *)
let at_w (i : Loop_index.t) =
  Expr.Coord.make ~n:(Loop_index.Const 0) ~t:(Loop_index.Const 0)
    ~d:(Loop_index.Const 0) ~h:(Loop_index.Const 0) ~w:i ~c:(Loop_index.Const 0)

let v n = Loop_var.of_int n
let temp n = Loop_temp.of_int n

let program ?(buffers = []) body =
  {
    Loop_program.buffers;
    body;
    scan_limits = Expr.Scan_limits.default;
    max_depth = 64;
  }

let bind_none (_ : Tensor_id.t) : Tensor.packed option = None
let f32_tensor shape f = Tensor.materialize shape f

let run_ok ?counters p ~bind =
  Err.or_raise ~pp_error:Loop_interp.pp_error
    (Loop_interp.run ?counters p ~bind)

let cells t n =
  List.init n (fun w -> Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w ~c:0))

(* [Kernel.create] on a one-value pixel kernel: input t0 (Caller), value t1 =
   [body], output t1. *)
let pixel_kernel ?(shape = shape_w 4) body =
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation = Region_group.Ref.Solo (Region_program.pixel body);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let load_t0 =
  Expr.Value.load
    (Expr_bridge.source_of_id (tid 0))
    (Expr_bridge.coord_of_vec6 Symbolic.out_vec)

let region_kernel =
  let local =
    Region_local.scalar
      ~id:(Expr.Builder.run Expr.Builder.fresh_local)
      ~value:(Expr.Value.const 3.)
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.create ~max_size:64 ~max_depth:16
         ~partition:
           (Err.or_raise ~pp_error:Region_partition.pp_error
              (Region_partition.of_whole_axes Expr.Axis.all))
         ~locals:[ local ]
         ~output:(Expr.Value.local local.Region_local.id))
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs:[]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 (shape_w 4) f32;
             computation = Region_group.Ref.Solo program;
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let i64_kernel =
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs:[]
       ~values_i64:
         [
           {
             Kernel.Value_i64.id = tid 1;
             sg = sg 1 (shape_w 2) (Payload.Fmt Payload.I64);
             pixel = Expr.Value.i64_const 7L;
           };
         ]
       ~values:[] ~outputs:[] ())

(* Two emitters sharing one local: a grouped run. *)
let grouped_kernel =
  let out_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1 in
  let local =
    Region_local.scalar
      ~id:(Expr.Builder.run Expr.Builder.fresh_local)
      ~value:(Expr.Value.const 1.)
  in
  let emitter output : Region_group.Emitter.t =
    {
      output_shape = out_shape;
      partition =
        Err.or_raise ~pp_error:Region_partition.pp_error
          (Region_partition.of_whole_axes [ Expr.Axis.H ]);
      key_axes = [ (Expr.Axis.W, Expr.Axis.W) ];
      output;
    }
  in
  let read = Expr.Value.local local.Region_local.id in
  let group =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.create ~max_size:64 ~max_depth:16
         ~canonical_shape:(shape_w 4) ~locals:[ local ]
         ~emitters:[ emitter read; emitter (Expr.Value.add read read) ])
  in
  let value id ordinal =
    {
      Kernel.Value.id = tid id;
      sg = sg id out_shape f32;
      computation =
        Region_group.Ref.Grouped (group, Region_group.Ordinal.of_int ordinal);
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs:[]
       ~values:[ value 1 0; value 2 1 ]
       ~outputs:[ tid 1; tid 2 ]
       ())
