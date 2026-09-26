open Loop_ir
open Loop_fixtures

(* Hand-lowered programs shared by the harness tests and the generated-code gate. *)

let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input
let output = buffer 1 (shape_w 4) f32 Loop_buffer.Output
let w = Loop_index.Var (v 0)

(* out[w] = round_f32(in[w] * 2.), the hand-lowered form of the pixel kernel. *)
let doubling =
  program ~buffers:[ input; output ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 4;
          body =
            [
              Loop_stmt.Store
                {
                  buffer = output;
                  coord = at_w w;
                  value =
                    Loop_stored.F32
                      (Loop_expr.Round_f32
                         (Loop_expr.Binary
                            ( Expr.Value.Mul,
                              Loop_expr.Load (input, at_w w),
                              Loop_expr.Const 2. )));
                };
            ];
        };
    ]

(* out[@f] = round_f32(in[@5 - f] * 2.) over a [H=2 W=3] pair: flat addressing
   ([Loop_expr.Load_flat], [Loop_stmt.Store_flat]), reversed so every offset
   peels into a different coordinate than it was written at. *)
let shape_hw = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1
let flat_input = buffer 0 shape_hw f32 Loop_buffer.Input
let flat_output = buffer 1 shape_hw f32 Loop_buffer.Output

let reversed_flat =
  program
    ~buffers:[ flat_input; flat_output ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 6;
          body =
            [
              Loop_stmt.Store_flat
                {
                  buffer = flat_output;
                  offset = w;
                  value =
                    Loop_stored.F32
                      (Loop_expr.Round_f32
                         (Loop_expr.Binary
                            ( Expr.Value.Mul,
                              Loop_expr.Load_flat
                                ( flat_input,
                                  Loop_index.Add
                                    ( Loop_index.Const 5,
                                      Loop_index.Scale (-1, w) ) ),
                              Loop_expr.Const 2. )));
                };
            ];
        };
    ]

let kernel = pixel_kernel (Expr.Value.mul load_t0 (Expr.Value.const 2.))
let plan = Fusion_plan.default kernel
let data = [| -0.; 1.5; nan; 3. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor (shape_w 4) (fun c -> data.(Dim.to_int (Vec6.get c Axis.W))))
  else None

let shifted_kernel =
  let w1 =
    Expr.Index.assume_position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output Expr.Axis.W))
         (Expr.Index.const 1))
  in
  pixel_kernel
    (Expr.Value.load
       (Expr_bridge.source_of_id (tid 0))
       (Expr.Coord.set
          (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
          Expr.Axis.W w1))

let shifted_loop ~extent =
  let shifted = Loop_index.Add (w, Loop_index.Const 1) in
  program ~buffers:[ input; output ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 4;
          body =
            [
              Loop_stmt.Fail_if
                ( Loop_bool.Out_of_range (shifted, extent),
                  Loop_failure.Load_out_of_range
                    { buffer = input; coord = at_w shifted } );
            ];
        };
    ]

let out_coord = Expr_bridge.coord_of_vec6 Symbolic.out_vec
let at axis i = Expr.Coord.set out_coord axis i
let ld ?(id = 0) c = Expr.Value.load (Expr_bridge.source_of_id (tid id)) c
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

(* A Filled input is materialized as f32 by the reference, so the fold must be
   the ROUNDED value: 2^24 + 1 is not representable and reads back as 2^24, and
   only adding 1 tells that apart (2^24 + 1 rounds down again, 2^24 + 2 does not). *)
let filled_kernel v =
  let shape = shape_w 4 in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
           {
             Kernel.Input.id = tid 2;
             sg = sg 2 shape f32;
             binding = Kernel.Binding.Filled v;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (Expr.Value.add load_t0 (ld ~id:2 out_coord)));
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let reduce_c kind ~lo ~hi =
  Expr.Builder.run
    (Expr.Builder.reduction ~kind ~lo ~hi (fun i ->
         Expr.Builder.return
           (ld
              (at Expr.Axis.C
                 (Expr.Index.of_position i |> Expr.Index.assume_position)))))

(* out[c] = reduce over C in [0, 3) of the input, at the single output cell. *)
let reduction_kernel kind ~lo ~hi =
  let in_shape = s1c 3 and out_shape = s1c 1 in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 in_shape f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 out_shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel (reduce_c kind ~lo ~hi));
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let hw h w = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1

let pool_kernel ~input ~out ~kernel ~stride ~pad ~result =
  let sq f v = Op_config.Hw.{ h = f v; w = f v } in
  let body =
    Expr.Value.intrinsic
      (Expr.Intrinsic.max_pool
         ~source:(Expr_bridge.source_of_id (tid 0))
         ~input:(sq Dim.extent input) ~kernel:(sq Dim.extent kernel)
         ~stride:(sq Op_config.Pos.of_int stride)
         ~pad:(sq Op_config.Nonneg.of_int pad)
         ~out:out_coord ~result)
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 (hw input input) f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 (hw out out) f32;
             computation = Region_group.Ref.Solo (Region_program.pixel body);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

(* 2^30 * w leaves the index domain at w = 2. *)
let overflow_kernel =
  pixel_kernel
    (Expr.Value.value_of_index
       (Expr.Index.scale (1 lsl 30)
          (Expr.Index.of_position (Expr.Index.output Expr.Axis.W))))

let two24 = 16777216.

(* t1 = t0 + 1 and t2 = t1 + 1, each rounded to f32 at its own boundary. With
   t0 = 2^24, t1 is 2^24 + 1 = not representable, so it reads back as 2^24 and
   t2 is 2^24 again; without the inner round t2 would be 2^24 + 2. *)
let chain ~outputs =
  let shape = shape_w 2 in
  let ld id =
    Expr.Value.load
      (Expr_bridge.source_of_id (tid id))
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  let plus_one id = Expr.Value.add (ld id) (Expr.Value.const 1.) in
  let value id body =
    {
      Kernel.Value.id = tid id;
      sg = sg id shape f32;
      computation = Region_group.Ref.Solo (Region_program.pixel body);
      result = Kernel.Result_conversion.Round_f32;
    }
  in
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
       ~values:[ value 1 (plus_one 0); value 2 (plus_one 1) ]
       ~outputs ())

(* [op] applied to the input at every cell. *)
let unary_kernel (op : Expr.Value.unary_op) =
  let apply =
    match op with
    | Expr.Value.Cos -> Expr.Value.cos
    | Expr.Value.Erf -> Expr.Value.erf
    | Expr.Value.Exp -> Expr.Value.exp
    | Expr.Value.Log -> Expr.Value.log
    | Expr.Value.Sin -> Expr.Value.sin
    | Expr.Value.Sqrt -> Expr.Value.sqrt
    | Expr.Value.Trunc -> Expr.Value.trunc
  in
  pixel_kernel (apply load_t0)

let full = (Expr.Index.zero, Expr.Index.const 3)

(* ---- Region programs ------------------------------------------------------- *)

let rows_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:3

let whole axes =
  Err.or_raise ~pp_error:Region_partition.pp_error
    (Region_partition.of_whole_axes axes)

let region_kernel_of ?(shape = rows_shape) program =
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
             computation = Region_group.Ref.Solo program;
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let region_finish ?(partition = whole [ Expr.Axis.C ]) output =
  Region_program.Builder.finish ~max_size:256 ~max_depth:32 ~partition ~output

let build program =
  Err.or_raise ~pp_error:Region_program.pp_error
    (Region_program.Builder.run program)

let row_sum =
  Expr.Builder.run
    (Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
       ~hi:(Expr.Index.const 3) (fun i ->
         Expr.Builder.return
           (ld
              (at Expr.Axis.C
                 (Expr.Index.assume_position (Expr.Index.of_position i))))))

(* out = x - sum(row): one scalar local read by every output of the key. *)
let centered_program =
  build
    (Region_program.Builder.scalar row_sum (fun total ->
         region_finish (Expr.Value.sub load_t0 total)))

(* A vector local of [extent] slots, each the input at its own position doubled;
   the emitter reads position [pick] of it. *)
let vector_program ~extent ~pick =
  build
    (Region_program.Builder.vector ~extent:(Slot.extent extent)
       (fun i ->
         Expr.Builder.return
           (Expr.Value.mul (ld (at Expr.Axis.C i)) (Expr.Value.const 2.)))
       (fun read -> region_finish (Expr.Value.add (read pick) load_t0)))

let pick i = Expr.Index.assume_position (Expr.Index.const i)

(* Every axis Whole: no Singleton axis, so a single key and one emitter loop over
   the whole tensor. The local reads no Output axis, as [check] requires. *)
let whole_only_program =
  build
    (Region_program.Builder.scalar (Expr.Value.const 3.) (fun three ->
         region_finish ~partition:(whole Expr.Axis.all)
           (Expr.Value.sub load_t0 three)))

(* ---- scans ------------------------------------------------------------------ *)

let scan_limits = Expr.Scan_limits.default

(* trace[0, l] = x[l]; trace[s+1, l] = trace[s, l] + x[l]: after [steps] updates a
   lane holds (steps + 1) * x[l]. *)
let scan_builder ~width ~steps continue =
  Region_program.Builder.scan ~limits:scan_limits ~width ~steps
    ~init:(fun ~lane -> Expr.Builder.return (ld (at Expr.Axis.C lane)))
    ~update:(fun ~step:_ ~lane ~previous_at ->
      Expr.Builder.return
        (Expr.Value.add (previous_at lane) (ld (at Expr.Axis.C lane))))
    continue

(* A trace local read by the emitter at its last row, one lane per output C. *)
let trace_program_at ~steps ~row ~lane =
  build
    (scan_builder ~width:3 ~steps (fun read -> region_finish (read ~row ~lane)))

let trace_program ~steps =
  trace_program_at ~steps
    ~row:(Expr.Index.assume_position (Expr.Index.const steps))
    ~lane:(Expr.Index.output Expr.Axis.C)

(* The same recurrence, inline and re-executed at every cell, in a Pixel body. *)
let inline_scan ~width ~steps =
  Err.or_raise ~pp_error:Expr.Scan.pp_error
    (Expr.Builder.run
       (Expr.Builder.scan ~limits:scan_limits ~width ~steps
          ~init:(fun ~lane -> Expr.Builder.return (ld (at Expr.Axis.C lane)))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Expr.Builder.return
              (Expr.Value.add (previous_at lane) (ld (at Expr.Axis.C lane))))))

let inline_scan_body_at ~steps ~row ~lane =
  Expr.Value.scan_at (inline_scan ~width:3 ~steps) ~row ~lane

let inline_scan_body ~steps =
  inline_scan_body_at ~steps
    ~row:(Expr.Index.assume_position (Expr.Index.const steps))
    ~lane:(Expr.Index.output Expr.Axis.C)

let inline_scan_kernel ~steps =
  region_kernel_of (Region_program.pixel (inline_scan_body ~steps))

(* The same kernels under scan limits of their own. *)
let with_scan_limits ~max_state ~max_updates program_of_limits =
  let limits =
    let d = Kernel.Limits.default in
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:d.Kernel.Limits.max_size
         ~max_depth:d.Kernel.Limits.max_depth
         ~max_values:d.Kernel.Limits.max_values
         ~max_dep_depth:d.Kernel.Limits.max_dep_depth
         ~max_inputs:d.Kernel.Limits.max_inputs
         ~max_outputs:d.Kernel.Limits.max_outputs
         ~max_extent:d.Kernel.Limits.max_extent
         ~max_numel:d.Kernel.Limits.max_numel
         ~max_bytes:d.Kernel.Limits.max_bytes
         ~max_local_slots:d.Kernel.Limits.max_local_slots
         ~max_scan_state:max_state ~max_scan_updates_per_key:max_updates
         ~max_scan_updates_total:d.Kernel.Limits.max_scan_updates_total)
  in
  program_of_limits limits

let limited_kernel ~limits computation =
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~limits
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 rows_shape f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 rows_shape f32;
             computation;
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

(* ---- Region groups ------------------------------------------------------------ *)

(* Two emitters share one local (W + 1, so 1..4) and one evaluation. The F32
   member stores it rounded; the second member stores [Nonzero_bool] of its own
   body, which for [second] = [read * 1e-50] lies below binary32's smallest
   subnormal: rounding to f32 first reads it as false, [Nonzero_bool] on the
   working value as true. *)
let bool_group_kernel ~second =
  let canonical_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:1 in
  let out_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1 in
  let local =
    Region_local.scalar
      ~id:(Expr.Builder.run Expr.Builder.fresh_local)
      ~value:
        (Expr.Value.add
           (Expr.Value.value_of_index
              (Expr.Index.of_position (Expr.Index.output Expr.Axis.W)))
           (Expr.Value.const 1.))
  in
  let read = Expr.Value.local local.Region_local.id in
  let emitter output : Region_group.Emitter.t =
    {
      output_shape = out_shape;
      partition = whole [ Expr.Axis.H ];
      key_axes = [ (Expr.Axis.W, Expr.Axis.W) ];
      output;
    }
  in
  let group =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.create ~max_size:64 ~max_depth:16 ~canonical_shape
         ~locals:[ local ]
         ~emitters:[ emitter read; emitter (second read) ])
  in
  let value id ordinal fmt result =
    {
      Kernel.Value.id = tid id;
      sg = sg id out_shape fmt;
      computation =
        Region_group.Ref.Grouped (group, Region_group.Ordinal.of_int ordinal);
      result;
    }
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs:[]
       ~values:
         [
           value 1 0 f32 Kernel.Result_conversion.Round_f32;
           value 2 1 (Payload.Fmt Payload.Bool)
             Kernel.Result_conversion.Nonzero_bool;
         ]
       ~outputs:[ tid 1; tid 2 ]
       ())

(* An int64 value that reads an F32 input as if it were int64: lowering refuses it
   by the source's format, and the reference fails at run time. *)
let i64_load_of_f32_kernel =
  let shape = shape_w 2 in
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
       ~values_i64:
         [
           {
             Kernel.Value_i64.id = tid 1;
             sg = sg 1 shape (Payload.Fmt Payload.I64);
             pixel =
               Expr.Value.i64_load
                 (Expr_bridge.source_of_id (tid 0))
                 (Expr_bridge.coord_of_vec6 Symbolic.out_vec);
           };
         ]
       ~values:[] ~outputs:[] ())

(* ---- int64 ---------------------------------------------------------------- *)

let i64 = Payload.Fmt Payload.I64

(* Inputs: t0 = F32 floats, t2 = I64 cells. One int64 value t1 over [shape_w 4]. *)
let i64_body_kernel body =
  let shape = shape_w 4 in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
           {
             Kernel.Input.id = tid 2;
             sg = sg 2 shape i64;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values_i64:
         [ { Kernel.Value_i64.id = tid 1; sg = sg 1 shape i64; pixel = body } ]
       ~values:[] ~outputs:[] ())

let ld_i64 ?(id = 2) coord =
  Expr.Value.i64_load (Expr_bridge.source_of_id (tid id)) coord

let i64_here = ld_i64 out_coord

let i64_bind ~floats ~cells id =
  let shape = shape_w 4 in
  if Tensor_id.equal id (tid 0) then
    Some (f32_tensor shape (fun c -> floats.((Vec6.offset shape c :> int))))
  else if Tensor_id.equal id (tid 2) then
    Some
      (Tensor.materialize_i64 shape (fun c ->
           cells.((Vec6.offset shape c :> int))))
  else None

(* ---- formats -------------------------------------------------------------- *)

let shape_c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

(* One F32 value that loads an input of any format and quantization at the same
   coordinate: every decode is a load's decode. *)
let format_kernel ~fmt ?quant n =
  let shape = shape_c n in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = Tensor_sig.create ~id:(tid 0) ~name:"" ~shape ~fmt ?quant ();
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation = Region_group.Ref.Solo (Region_program.pixel load_t0);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

(* A tensor of raw storage cells in the given format: [cells] are the storage
   integers, and for F32/F64 the bit patterns of the floats. *)
let raw_tensor (Payload.Fmt fmt) ?quant ?shape n (cells : int64 array) :
    Tensor.packed =
  let shape = Option.value shape ~default:(shape_c n) in
  let fill kind f =
    let data = Bigarray.Array1.create kind Bigarray.c_layout n in
    Array.iteri (fun i c -> data.{i} <- f c) cells;
    data
  in
  let quantized () =
    match quant with
    | Some q -> Payload.Quant q
    | None -> invalid_arg "raw_tensor: a quantized format needs parameters"
  in
  match fmt with
  | Payload.BF16 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.int16_unsigned Int64.to_int;
            };
        }
  | Payload.Bool ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.int8_unsigned Int64.to_int;
            };
        }
  | Payload.F16 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.int16_unsigned Int64.to_int;
            };
        }
  | Payload.F32 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.float32 Int64.float_of_bits;
            };
        }
  | Payload.F64 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.float64 Int64.float_of_bits;
            };
        }
  | Payload.I16 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = quantized ();
              data = fill Bigarray.int16_signed Int64.to_int;
            };
        }
  | Payload.I32 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.int32 Int64.to_int32;
            };
        }
  | Payload.I64 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = Payload.No_quant;
              data = fill Bigarray.int64 Fun.id;
            };
        }
  | Payload.I8 ->
      Tensor.Tensor
        {
          shape;
          payload =
            {
              Payload.fmt;
              quant = quantized ();
              data = fill Bigarray.int8_signed Int64.to_int;
            };
        }

(* t1 = t0 * 2 (f32); t3 = int64 (float_to_i64 t1) + t2; t4 = float of t3, plus 1.
   The two lists are one dependency graph: t3 reads a float value and t4 reads
   t3, so the order is t1, t3, t4. *)
let mixed_kernel =
  let shape = shape_w 4 in
  let ld_f id = Expr.Value.load (Expr_bridge.source_of_id (tid id)) out_coord in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
           {
             Kernel.Input.id = tid 2;
             sg = sg 2 shape i64;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values_i64:
         [
           {
             Kernel.Value_i64.id = tid 3;
             sg = sg 3 shape i64;
             pixel =
               Expr.Value.i64_add
                 (Expr.Value.float_to_i64 (ld_f 1))
                 (ld_i64 out_coord);
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (Expr.Value.mul (ld_f 0) (Expr.Value.const 2.)));
             result = Kernel.Result_conversion.Round_f32;
           };
           {
             Kernel.Value.id = tid 4;
             sg = sg 4 shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (Expr.Value.add
                       (Expr.Value.i64_to_float
                          (Expr.Value.i64_load
                             (Expr_bridge.source_of_id (tid 3))
                             out_coord))
                       (Expr.Value.const 1.)));
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1; tid 4 ]
       ())

(* A Bool value from a float body, read by a float value: t1 = nonzero t0; t2 = t1 + 1.
   [input_fmt] is t0's storage, F64 to reach values binary32 cannot hold. *)
let bool_kernel ~input_fmt =
  let shape = shape_w 4 in
  let ld id = Expr.Value.load (Expr_bridge.source_of_id (tid id)) out_coord in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg =
               Tensor_sig.create ~id:(tid 0) ~name:"" ~shape ~fmt:input_fmt ();
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape (Payload.Fmt Payload.Bool);
             computation = Region_group.Ref.Solo (Region_program.pixel (ld 0));
             result = Kernel.Result_conversion.Nonzero_bool;
           };
           {
             Kernel.Value.id = tid 2;
             sg = sg 2 shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (Expr.Value.add (ld 1) (Expr.Value.const 1.)));
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 2 ]
       ())

(* A Bool input the kernel fills itself, read by a float value. *)
let bool_fill_kernel v =
  let shape = shape_w 4 in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape (Payload.Fmt Payload.Bool);
             binding = Kernel.Binding.Filled v;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation =
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (Expr.Value.add load_t0 (Expr.Value.const 1.)));
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let reduce_w kind ?(lo = Expr.Index.zero) ?(hi = Expr.Index.const 4) () =
  Expr.Builder.run
    (Expr.Builder.i64_reduction ~kind ~lo ~hi (fun i ->
         Expr.Builder.return
           (ld_i64
              (Expr.Coord.set out_coord Expr.Axis.W
                 (Expr.Index.assume_position (Expr.Index.of_position i))))))

(* out[w] = t0[w := t2[w]]: the index component is the stored value of an int64
   tensor, checked against [-extent, extent - 1] and normalized. *)
let gather_kernel =
  let shape = shape_w 4 in
  let src = Expr_bridge.source_of_id (tid 2) in
  let gathered =
    Expr.Value.load
      (Expr_bridge.source_of_id (tid 0))
      (Expr.Coord.set out_coord Expr.Axis.W (Expr.Index.data src out_coord 4))
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
           {
             Kernel.Input.id = tid 2;
             sg = sg 2 shape i64;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape f32;
             computation = Region_group.Ref.Solo (Region_program.pixel gathered);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 1 ]
       ())
