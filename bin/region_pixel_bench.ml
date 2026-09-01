(* Reproducible Pixel-loop benchmark for Region Foundation.  This deliberately
   builds a [Region_program.pixel] Kernel and calls [Kernel_eval.run], rather
   than timing [Expr.Eval] in isolation: wrapper classification, result
   conversion and Tensor materialization are part of the hot path we must keep
   stable. *)

let samples = 20
let warmup = 3
let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:32 ~c:64
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int

let signature ~shape ~fmt id =
  Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ()

let coord = Expr_bridge.coord_of_vec6 Symbolic.out_vec
let load id = Expr.Value.load (Expr_bridge.source_of_id (tid id)) coord

let value ~shape id body =
  {
    Kernel.Value.id = tid id;
    sg = signature ~shape ~fmt:f32 id;
    computation = Region_program.pixel body;
    result = Kernel.Result_conversion.Round_f32;
  }

let input ~shape ?(fmt = f32) id =
  {
    Kernel.Input.id = tid id;
    sg = signature ~shape ~fmt id;
    binding = Kernel.Binding.Caller;
  }

let kernel ~output_shape ~inputs body =
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs
       ~values:[ value ~shape:output_shape (List.length inputs) body ]
       ~outputs:[ tid (List.length inputs) ]
       ())

let tensor shape value = Tensor.materialize shape (fun _ -> value)

let median xs =
  let ys = Array.of_list xs in
  Array.sort Float.compare ys;
  ys.(Array.length ys / 2)

let words () =
  let s = Gc.quick_stat () in
  s.Gc.minor_words +. s.Gc.major_words

let bench name ~loads kernel bind =
  for _ = 1 to warmup do
    ignore
      (Err.or_raise ~pp_error:Kernel_eval.pp_error
         (Kernel_eval.run kernel ~bind))
  done;
  let timings, allocated =
    List.init samples (fun _ ->
        Gc.full_major ();
        let before = words () and started = Unix.gettimeofday () in
        ignore
          (Err.or_raise ~pp_error:Kernel_eval.pp_error
             (Kernel_eval.run kernel ~bind));
        ((Unix.gettimeofday () -. started) *. 1000., words () -. before))
    |> List.split
  in
  let cells = (Vec6.numel shape :> int) in
  Printf.printf
    "%s samples=%d median_ms=%.3f gc_words_per_output=%.3f output_cells=%d \
     input_loads=%d\n\
     %!"
    name samples (median timings)
    (median allocated /. float_of_int cells)
    cells (cells * loads)

let last_coord shape =
  Vec6.coord
    ~n:(Dim.to_int (Vec6.get shape Axis.N) - 1)
    ~t:(Dim.to_int (Vec6.get shape Axis.T) - 1)
    ~d:(Dim.to_int (Vec6.get shape Axis.D) - 1)
    ~h:(Dim.to_int (Vec6.get shape Axis.H) - 1)
    ~w:(Dim.to_int (Vec6.get shape Axis.W) - 1)
    ~c:(Dim.to_int (Vec6.get shape Axis.C) - 1)

let smoke name ~output_shape ~output_id ~loads kernel bind ~expected_first
    ~expected_last =
  let output =
    Tensor_id.Map.find (tid output_id)
      (Err.or_raise ~pp_error:Kernel_eval.pp_error
         (Kernel_eval.run kernel ~bind))
  in
  if
    (not (Float.equal (Tensor.read output Vec6.origin) expected_first))
    || not
         (Float.equal
            (Tensor.read output (last_coord output_shape))
            expected_last)
  then failwith (Printf.sprintf "%s smoke result differs" name);
  Printf.printf "%s_smoke output_cells=%d input_loads=%d\n%!" name
    (Vec6.numel output_shape :> int)
    ((Vec6.numel output_shape :> int) * loads)

let shifted axis amount =
  Expr.Index.assume_position
    (Expr.Index.add
       (Expr.Index.of_position (Expr.Index.output axis))
       (Expr.Index.const amount))

let conv_coord dh dw =
  Expr.Coord.make ~n:(Expr.Index.output Axis.N) ~t:(Expr.Index.output Axis.T)
    ~d:(Expr.Index.output Axis.D) ~h:(shifted Axis.H dh) ~w:(shifted Axis.W dw)
    ~c:(Expr.Index.output Axis.C)

let conv_smoke () =
  let output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:30 ~w:30 ~c:8 in
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:32 ~w:32 ~c:8 in
  let source = Expr_bridge.source_of_id (tid 0) in
  let body =
    List.concat_map (fun h -> List.init 3 (fun w -> (h, w))) [ 0; 1; 2 ]
    |> List.map (fun (h, w) -> Expr.Value.load source (conv_coord h w))
    |> List.fold_left Expr.Value.add (Expr.Value.const 0.)
  in
  let kernel =
    kernel ~output_shape ~inputs:[ input ~shape:input_shape 0 ] body
  in
  smoke "conv" ~output_shape ~output_id:1 ~loads:9 kernel
    (fun id ->
      if Tensor_id.equal id (tid 0) then
        Some
          (Tensor.materialize input_shape (fun c ->
               float_of_int
                 ((100 * Dim.to_int (Vec6.get c Axis.H))
                 + (10 * Dim.to_int (Vec6.get c Axis.W))
                 + Dim.to_int (Vec6.get c Axis.C))))
      else None)
    ~expected_first:990. ~expected_last:29763.

let index_smoke () =
  let output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:64 in
  let index_source = Expr_bridge.source_of_id (tid 0) in
  let value_source = Expr_bridge.source_of_id (tid 1) in
  let gathered =
    Expr.Index.data index_source coord
      (Dim.to_int (Vec6.get output_shape Axis.C))
  in
  let body =
    Expr.Value.load value_source
      (Expr.Coord.make ~n:(Expr.Index.output Axis.N)
         ~t:(Expr.Index.output Axis.T) ~d:(Expr.Index.output Axis.D)
         ~h:(Expr.Index.output Axis.H) ~w:(Expr.Index.output Axis.W) ~c:gathered)
  in
  let kernel =
    kernel ~output_shape
      ~inputs:
        [
          input ~shape:output_shape ~fmt:(Payload.Fmt Payload.I64) 0;
          input ~shape:output_shape 1;
        ]
      body
  in
  let indices =
    Tensor.materialize_i64 output_shape (fun c ->
        Int64.of_int (63 - Dim.to_int (Vec6.get c Axis.C)))
  in
  smoke "index" ~output_shape ~output_id:2 ~loads:1 kernel
    (fun id ->
      if Tensor_id.equal id (tid 0) then Some indices
      else if Tensor_id.equal id (tid 1) then
        Some
          (Tensor.materialize output_shape (fun c ->
               float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1)))
      else None)
    ~expected_first:64. ~expected_last:1.

let () =
  let one = tensor shape 1. and two = tensor shape 2. in
  let identity =
    kernel ~output_shape:shape ~inputs:[ input ~shape 0 ] (load 0)
  in
  bench "identity" ~loads:1 identity (fun id ->
      if Tensor_id.equal id (tid 0) then Some one else None);
  let add =
    kernel ~output_shape:shape
      ~inputs:[ input ~shape 0; input ~shape 1 ]
      (Expr.Value.add (load 0) (load 1))
  in
  bench "add" ~loads:2 add (fun id ->
      if Tensor_id.equal id (tid 0) then Some one
      else if Tensor_id.equal id (tid 1) then Some two
      else None);
  conv_smoke ();
  index_smoke ()
