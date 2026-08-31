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
let signature id = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt:f32 ()
let coord = Expr_bridge.coord_of_vec6 Symbolic.out_vec
let load id = Expr.Value.load (Expr_bridge.source_of_id (tid id)) coord

let value id body =
  {
    Kernel.Value.id = tid id;
    sg = signature id;
    computation = Region_program.pixel body;
    result = Kernel.Result_conversion.Round_f32;
  }

let input id =
  {
    Kernel.Input.id = tid id;
    sg = signature id;
    binding = Kernel.Binding.Caller;
  }

let kernel ~inputs body =
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create ~inputs
       ~values:[ value (List.length inputs) body ]
       ~outputs:[ tid (List.length inputs) ]
       ())

let tensor value = Tensor.materialize shape (fun _ -> value)

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

let () =
  let one = tensor 1. and two = tensor 2. in
  let identity = kernel ~inputs:[ input 0 ] (load 0) in
  bench "identity" ~loads:1 identity (fun id ->
      if Tensor_id.equal id (tid 0) then Some one else None);
  let add =
    kernel ~inputs:[ input 0; input 1 ] (Expr.Value.add (load 0) (load 1))
  in
  bench "add" ~loads:2 add (fun id ->
      if Tensor_id.equal id (tid 0) then Some one
      else if Tensor_id.equal id (tid 1) then Some two
      else None)
