open Ssa_bridge
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures

(* A [Contract] through generated C against the interpreter, which runs
   [Contraction.eval]: batched, strided and transposed operands, both binary32
   numerics, and a seed taken from the output's own cells. Every output cell
   must match bit for bit. *)

let lcg = ref 99

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let operand ~batch ~outer ~k = { Contraction.Operand.batch; outer; k }

let desc ~batch ~m ~n ~k ~a ~b ~c ~seed ~numerics =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch ~m ~n ~k ~a ~b ~c
          ~format:Contraction.Element.F32 ~seed ~numerics))

(* [passes] contractions into C, the second on seeing the first's cells *)
let run ~name ~na ~nb ~nc descs =
  let buffers =
    [
      Fx.buffer 0 ~h:1L ~w:(Int64.of_int na) Ssa_format.F32 Ssa_buffer.Input;
      Fx.buffer 1 ~h:1L ~w:(Int64.of_int nb) Ssa_format.F32 Ssa_buffer.Input;
      Fx.buffer 2 ~h:1L ~w:(Int64.of_int nc) Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  let p =
    Err.or_raise ~pp_error:Ssa_verify.pp_error
      (B.program ~buffers (fun bld ->
           List.iter
             (fun d ->
               B.contract bld d ~a:(Fx.buf 0) ~b:(Fx.buf 1) ~c:(Fx.buf 2))
             descs))
  in
  let a = Array.init na (fun _ -> value ())
  and b = Array.init nb (fun _ -> value ()) in
  let want = Array.make nc 0. in
  let memory =
    Fx.memory [ (0, Fx.floats a); (1, Fx.floats b); (2, Fx.floats want) ]
  in
  Fx.run p ~memory;
  let kernel, sites =
    match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_c.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p))
      []
  in
  let tensor id cells =
    if Tensor_id.equal id (Lf.tid 0) then
      Some
        (Lf.f32_tensor (Lf.shape_w na) (fun c ->
             cells.((Vec6.offset (Lf.shape_w na) c :> int))))
    else if Tensor_id.equal id (Lf.tid 1) then
      Some
        (Lf.f32_tensor (Lf.shape_w nb) (fun c ->
             b.((Vec6.offset (Lf.shape_w nb) c :> int))))
    else None
  in
  match
    Err.payload
      (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind:(fun id -> tensor id a))
  with
  | Error e -> Fmt.failwith "%a" Loop_c_exec.pp_error e
  | Ok outputs ->
      let got =
        match Tensor_id.Map.find_opt (Lf.tid 2) outputs with
        | Some t -> Array.of_list (Lf.cells t nc)
        | None -> Array.make nc 0.
      in
      let cells =
        match Ssa_id.Buffer.Map.find (Fx.buf 2) memory with
        | Ssa_memory.Floats cs -> cs
        | _ -> assert false
      in
      Fmt.pr "%s: C equals the interpreter: %b@." name
        (Array.for_all2 Core.Float_bits.equal_portable cells got);
      cells

let%expect_test "a contraction through C equals the interpreter" =
  let i = Int64.of_int in
  let go ?(batch = 1) ?(transposed = false) ~m ~n ~k numerics =
    let mk seed =
      desc ~batch:(i batch) ~m:(i m) ~n:(i n) ~k:(i k)
        ~a:(operand ~batch:(i (m * k)) ~outer:(i k) ~k:1L)
        ~b:
          (if transposed then operand ~batch:(i (k * n)) ~outer:(i k) ~k:1L
           else operand ~batch:(i (k * n)) ~outer:1L ~k:(i n))
        ~c:{ Contraction.Output.batch = i (m * n); row = i n; col = 1L }
        ~seed ~numerics
    in
    run
      ~name:
        (Fmt.str "%s %dx%dx%d b%d%s"
           (Contraction.Numerics.name numerics)
           m n k batch
           (if transposed then " Bt" else ""))
      ~na:(batch * m * k)
      ~nb:(batch * k * n)
      ~nc:(batch * m * n)
      [ mk Contraction.Seed.Zero; mk Contraction.Seed.Tensor ]
  in
  let ordered = go ~m:5 ~n:7 ~k:11 Contraction.Numerics.Ordered in
  let fused = go ~m:5 ~n:7 ~k:11 Contraction.Numerics.Fused_sequential in
  ignore (go ~batch:3 ~m:4 ~n:6 ~k:9 Contraction.Numerics.Ordered);
  ignore
    (go ~batch:2 ~transposed:true ~m:3 ~n:5 ~k:8
       Contraction.Numerics.Fused_sequential);
  ignore (go ~m:1 ~n:1 ~k:1 Contraction.Numerics.Ordered);
  (* the two numerics are not the same function, so the check can fail *)
  Fmt.pr "ordered and fused differ somewhere: %b@."
    (not (Array.for_all2 Core.Float_bits.equal_portable ordered fused));
  [%expect
    {|
    ordered 5x7x11 b1: C equals the interpreter: true
    fused_sequential 5x7x11 b1: C equals the interpreter: true
    ordered 4x6x9 b3: C equals the interpreter: true
    fused_sequential 3x5x8 b2 Bt: C equals the interpreter: true
    ordered 1x1x1 b1: C equals the interpreter: true
    ordered and fused differ somewhere: true |}]
