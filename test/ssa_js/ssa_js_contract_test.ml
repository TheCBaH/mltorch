open Ssa_bridge
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures

(* A [Contract] through generated JavaScript against the interpreter, which runs
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
let run ?inputs ~name ~na ~nb ~nc descs =
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
  let a, b =
    match inputs with
    | Some (a, b) -> (a, b)
    | None ->
        let a = Array.init na (fun _ -> value ()) in
        (a, Array.init nb (fun _ -> value ()))
  in
  let want = Array.make nc 0. in
  let memory =
    Fx.memory [ (0, Fx.floats a); (1, Fx.floats b); (2, Fx.floats want) ]
  in
  Fx.run p ~memory;
  let ast, sites =
    match Ssa_js.program p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_js.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_js.arguments p))
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
  let compiled =
    match
      Err.payload
        (Loop_js_exec.compile_kernel ~sites loop (Js_print.factory_body ast))
    with
    | Ok c -> c
    | Error (`Js_compile m) ->
        Fmt.failwith "the engine refused the source: %s" m
  in
  match
    Err.payload (Loop_js_exec.run compiled ~bind:(fun id -> tensor id a))
  with
  | Error e -> Fmt.failwith "%a" Loop_js_exec.pp_error e
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
      Fmt.pr "%s: JS equals the interpreter: %b@." name
        (Array.for_all2 Core.Float_bits.equal_portable cells got);
      cells

let%expect_test "a contraction through JavaScript equals the interpreter" =
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
  ignore (go ~batch:3 ~m:4 ~n:6 ~k:9 Contraction.Numerics.Ordered);
  ignore
    (go ~batch:2 ~transposed:true ~m:3 ~n:5 ~k:8 Contraction.Numerics.Ordered);
  ignore (go ~m:1 ~n:1 ~k:1 Contraction.Numerics.Ordered);
  ignore ordered;
  ignore (go ~m:5 ~n:7 ~k:11 Contraction.Numerics.Fused_sequential);
  ignore (go ~batch:3 ~m:4 ~n:6 ~k:9 Contraction.Numerics.Fused_sequential);
  ignore
    (go ~batch:2 ~transposed:true ~m:3 ~n:5 ~k:8
       Contraction.Numerics.Fused_sequential);
  ignore (go ~m:1 ~n:1 ~k:1 Contraction.Numerics.Fused_sequential);
  [%expect
    {|
    ordered 5x7x11 b1: JS equals the interpreter: true
    ordered 4x6x9 b3: JS equals the interpreter: true
    ordered 3x5x8 b2 Bt: JS equals the interpreter: true
    ordered 1x1x1 b1: JS equals the interpreter: true
    fused_sequential 5x7x11 b1: JS equals the interpreter: true
    fused_sequential 4x6x9 b3: JS equals the interpreter: true
    fused_sequential 3x5x8 b2 Bt: JS equals the interpreter: true
    fused_sequential 1x1x1 b1: JS equals the interpreter: true |}]

(* The case that separates a fused step from a product and an add: [m * m] with
   [m = 1 + 2^-12] is exactly halfway between two binary32 neighbours, and a
   tiny addend of either sign decides which way the one rounding goes. The
   addend is the first step's accumulator, so the second step is the fused one
   and carries the tie. *)
let%expect_test "the fused step is a single rounding in JavaScript" =
  let m = 1. +. ldexp 1. (-12) and tiny = ldexp 1. (-30) in
  List.iter
    (fun (name, a, b) ->
      let d =
        desc ~batch:1L ~m:1L ~n:1L ~k:2L
          ~a:(operand ~batch:0L ~outer:2L ~k:1L)
          ~b:(operand ~batch:0L ~outer:1L ~k:1L)
          ~c:{ Contraction.Output.batch = 0L; row = 1L; col = 1L }
          ~seed:Contraction.Seed.Zero
          ~numerics:Contraction.Numerics.Fused_sequential
      in
      ignore (run ~inputs:(a, b) ~name ~na:2 ~nb:2 ~nc:1 [ d ]))
    [
      ("tiny addend above", [| tiny; m |], [| tiny; m |]);
      ("tiny addend below", [| tiny; m |], [| -.tiny; m |]);
      ("negated, above", [| tiny; -.m |], [| -.tiny; m |]);
      ("negated, below", [| tiny; -.m |], [| tiny; m |]);
    ];
  [%expect
    {|
    tiny addend above: JS equals the interpreter: true
    tiny addend below: JS equals the interpreter: true
    negated, above: JS equals the interpreter: true
    negated, below: JS equals the interpreter: true |}]

(* The paired numerics and every other format are refused rather than computed
   differently. *)
let%expect_test "what a contraction in JavaScript refuses" =
  let refuse ~format ~numerics =
    let d =
      Result.get_ok
        (Err.payload
           (Contraction.make ~batch:1L ~m:2L ~n:2L ~k:2L
              ~a:(operand ~batch:0L ~outer:2L ~k:1L)
              ~b:(operand ~batch:0L ~outer:1L ~k:2L)
              ~c:{ Contraction.Output.batch = 0L; row = 2L; col = 1L }
              ~format ~seed:Contraction.Seed.Zero ~numerics))
    in
    let fmt_of = function
      | Contraction.Element.Bf16 -> Ssa_format.Bf16
      | Contraction.Element.F32 | Contraction.Element.I8 _ -> Ssa_format.F32
    in
    let buffers =
      [
        Fx.buffer 0 ~h:1L ~w:4L (fmt_of format) Ssa_buffer.Input;
        Fx.buffer 1 ~h:1L ~w:4L (fmt_of format) Ssa_buffer.Input;
        Fx.buffer 2 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Output;
      ]
    in
    let p =
      Err.or_raise ~pp_error:Ssa_verify.pp_error
        (B.program ~buffers (fun bld ->
             B.contract bld d ~a:(Fx.buf 0) ~b:(Fx.buf 1) ~c:(Fx.buf 2)))
    in
    match Ssa_js.program p with
    | Ok _ -> Fmt.pr "lowered@."
    | Error e -> Fmt.pr "%a@." Ssa_js.pp_error e
  in
  refuse ~format:Contraction.Element.F32
    ~numerics:Contraction.Numerics.Fused_sequential;
  refuse ~format:Contraction.Element.F32 ~numerics:Contraction.Numerics.Ordered;
  refuse ~format:Contraction.Element.Bf16 ~numerics:Contraction.Numerics.Paired;
  refuse ~format:Contraction.Element.Bf16 ~numerics:Contraction.Numerics.Ordered;
  [%expect
    {|
    lowered
    lowered
    a contraction over other than binary32 has no JavaScript form
    a contraction over other than binary32 has no JavaScript form |}]
