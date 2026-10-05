open Loop_ir_test

(* Kernels and bindings for the direct-lowering suites. The pointwise and
   reduction kernels are the Loop IR suite's own, so both executors and the
   reference see the same source. *)

let tid = Loop_fixtures.tid
let f32 = Loop_fixtures.f32
let sg = Loop_fixtures.sg
let hw h w = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1

let bind_data ~shape data id =
  if Tensor_id.equal id (tid 0) then
    Some
      (Loop_fixtures.f32_tensor shape (fun c ->
           data.((Vec6.offset shape c :> int))))
  else None

(* out[h,w] = round_f32 (sum over k in [0, K) of A[h,k] * B[k,w]), with A the
   input t0 of shape M x K, B the input t1 of K x N and the output t2. *)
let matmul_kernel ~m ~k ~n =
  let position i = Expr.Index.assume_position (Expr.Index.of_position i) in
  let body =
    Expr.Builder.run
      (Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
         ~hi:(Expr.Index.const k) (fun i ->
           Expr.Builder.return
             (Expr.Value.mul
                (Loop_programs.ld ~id:0
                   (Loop_programs.at Expr.Axis.W (position i)))
                (Loop_programs.ld ~id:1
                   (Loop_programs.at Expr.Axis.H (position i))))))
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 (hw m k) f32;
             binding = Kernel.Binding.Caller;
           };
           {
             Kernel.Input.id = tid 1;
             sg = sg 1 (hw k n) f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 2;
             sg = sg 2 (hw m n) f32;
             computation = Region_group.Ref.Solo (Region_program.pixel body);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ tid 2 ]
       ())

(* Deterministic operands that are exact in binary32 and span magnitudes, so
   a reordered or re-rounded sum shows. *)
let operand seed len =
  Array.init len (fun i ->
      let x = float_of_int (((i * 7) + seed) mod 23) -. 11. in
      Core.Float_bits.(ignore equal_portable);
      Int32.float_of_bits (Int32.bits_of_float (x *. 0.37)))

let matmul_bind ~m ~k ~n ~a ~b id =
  let tensor shape data =
    Some
      (Loop_fixtures.f32_tensor shape (fun c ->
           data.((Vec6.offset shape c :> int))))
  in
  if Tensor_id.equal id (tid 0) then tensor (hw m k) a
  else if Tensor_id.equal id (tid 1) then tensor (hw k n) b
  else None

(* A four-cell reduction over C into one cell. *)
let four_cell_reduction kind =
  let body =
    Expr.Builder.run
      (Expr.Builder.reduction ~kind ~lo:Expr.Index.zero ~hi:(Expr.Index.const 4)
         (fun i ->
           Expr.Builder.return
             (Loop_programs.ld
                (Loop_programs.at Expr.Axis.C
                   (Expr.Index.of_position i |> Expr.Index.assume_position)))))
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = Loop_fixtures.tid 0;
             sg = Loop_fixtures.sg 0 (Loop_programs.s1c 4) Loop_fixtures.f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = Loop_fixtures.tid 1;
             sg = Loop_fixtures.sg 1 (Loop_programs.s1c 1) Loop_fixtures.f32;
             computation = Region_group.Ref.Solo (Region_program.pixel body);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ Loop_fixtures.tid 1 ]
       ())

(* t1[w] = select (w < 2.) t0[w + 1] 0.: the untaken arm reads out of range at
   w = 3, so an eager select fails where the reference does not. *)
let lazy_select_kernel =
  let w = Expr.Index.output Expr.Axis.W in
  let shifted =
    Expr.Index.add (Expr.Index.of_position w) (Expr.Index.const 1)
  in
  Loop_fixtures.pixel_kernel
    (Expr.Value.select
       (Expr.Bool.value_lt
          (Expr.Value.value_of_index (Expr.Index.of_position w))
          (Expr.Value.const 2.))
       (Loop_programs.ld
          (Loop_programs.at Expr.Axis.W (Expr.Index.assume_position shifted)))
       (Expr.Value.const 0.))
