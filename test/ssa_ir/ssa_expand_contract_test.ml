(* A contraction expanded into loops computes, once narrowed to binary32 and
   interpreted, the bits the [Contract] itself computes ([Contraction.eval]).
   Whatever the pass does not expand stays a [Contract]. *)

open Ssa_ir
module F = Ssa_fixtures
module B = Ssa_builder
module A = Attention_block
module E = Ssa_expand_attend_test

let i64 = Int64.of_int
let operand ~batch ~outer ~k = { Contraction.Operand.batch; outer; k }

let desc ?(format = Contraction.Element.F32) ?(batch = 1) ?(transposed = false)
    ~m ~n ~k ~seed ~numerics () =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:(i64 batch) ~m:(i64 m) ~n:(i64 n) ~k:(i64 k)
          ~a:(operand ~batch:(i64 (m * k)) ~outer:(i64 k) ~k:1L)
          ~b:
            (if transposed then
               operand ~batch:(i64 (k * n)) ~outer:(i64 k) ~k:1L
             else operand ~batch:(i64 (k * n)) ~outer:1L ~k:(i64 n))
          ~c:{ Contraction.Output.batch = i64 (m * n); row = i64 n; col = 1L }
          ~format ~seed ~numerics))

let count_contracts p =
  let n = ref 0 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr { op = Ssa_op.Contract _; _ } -> incr n
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For f -> region f.body
    | Ssa_stmt.If f ->
        region f.then_;
        region f.else_
    | Ssa_stmt.Ordered_sum f -> region f.body
  in
  region p.Ssa_program.entry;
  !n

let lcg = ref 7

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let run p ~a_cells ~b_cells ~nc =
  let memory =
    F.memory
      [
        (0, a_cells);
        (1, b_cells);
        (2, F.floats (Array.init nc (fun i -> float_of_int i /. 8.)));
      ]
  in
  F.run p ~memory;
  match Ssa_id.Buffer.Map.find (F.buf 2) memory with
  | Ssa_memory.Floats cs -> cs
  | _ -> assert false

let case ~name d =
  let format_buffer id n role =
    let fmt =
      match d.Contraction.format with
      | Contraction.Element.Bf16 when id < 2 -> Ssa_format.Bf16
      | _ -> Ssa_format.F32
    in
    F.buffer id ~h:1L ~w:(i64 n) fmt role
  in
  let na =
    Int64.to_int
      (Int64.mul d.Contraction.batch
         (Int64.mul d.Contraction.m d.Contraction.k))
  and nb =
    Int64.to_int
      (Int64.mul d.Contraction.batch
         (Int64.mul d.Contraction.k d.Contraction.n))
  and nc =
    Int64.to_int
      (Int64.mul d.Contraction.batch
         (Int64.mul d.Contraction.m d.Contraction.n))
  in
  let buffers =
    [
      format_buffer 0 na Ssa_buffer.Input;
      format_buffer 1 nb Ssa_buffer.Input;
      format_buffer 2 nc Ssa_buffer.Output;
    ]
  in
  let p =
    F.build ~buffers (fun bld ->
        B.contract bld d ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2))
  in
  let cells n =
    match d.Contraction.format with
    | Contraction.Element.Bf16 ->
        Ssa_memory.Ints
          (Array.init n (fun _ ->
               Int32.to_int
                 (Int32.shift_right_logical
                    (Int32.bits_of_float (A.bf16_round (value ())))
                    16)))
    | _ -> F.floats (Array.init n (fun _ -> value ()))
  in
  let a_cells = cells na and b_cells = cells nb in
  let expanded = Ssa_expand_contract.program p in
  match d.Contraction.numerics with
  | Contraction.Numerics.Fused_sequential ->
      (* the same loops; the plan's dot fusion makes them fused, which the
         structured interpreter does not run (the machine model's tests do) *)
      Fmt.pr "%s: %d contracts left; the plan fuses the sums@." name
        (count_contracts expanded)
  | Contraction.Numerics.Ordered | Contraction.Numerics.Paired ->
      let want = run p ~a_cells ~b_cells ~nc in
      let got =
        run (Ssa_precision.to_f32 ~exact:true expanded) ~a_cells ~b_cells ~nc
      in
      Fmt.pr "%s: %d contracts left; %s@." name (count_contracts expanded)
        (if
           Array.for_all2
             (fun a b -> Int32.bits_of_float a = Int32.bits_of_float b)
             want got
         then "equals Contraction.eval"
         else "DIFFERS from Contraction.eval")

let%expect_test "an expanded contraction equals the Contract" =
  let ordered = Contraction.Numerics.Ordered in
  let zero = Contraction.Seed.Zero and tensor = Contraction.Seed.Tensor in
  case ~name:"f32 5x7x11 zero"
    (desc ~m:5 ~n:7 ~k:11 ~seed:zero ~numerics:ordered ());
  case ~name:"f32 5x7x11 tensor"
    (desc ~m:5 ~n:7 ~k:11 ~seed:tensor ~numerics:ordered ());
  case ~name:"f32 b3 4x6x9"
    (desc ~batch:3 ~m:4 ~n:6 ~k:9 ~seed:tensor ~numerics:ordered ());
  case ~name:"f32 b2 Bt 3x5x8"
    (desc ~batch:2 ~transposed:true ~m:3 ~n:5 ~k:8 ~seed:zero ~numerics:ordered
       ());
  case ~name:"bf16 4x5x6"
    (desc ~format:Contraction.Element.Bf16 ~m:4 ~n:5 ~k:6 ~seed:zero
       ~numerics:ordered ());
  case ~name:"f32 fused"
    (desc ~m:3 ~n:3 ~k:4 ~seed:zero
       ~numerics:Contraction.Numerics.Fused_sequential ());
  [%expect
    {|
    f32 5x7x11 zero: 0 contracts left; equals Contraction.eval
    f32 5x7x11 tensor: 0 contracts left; equals Contraction.eval
    f32 b3 4x6x9: 0 contracts left; equals Contraction.eval
    f32 b2 Bt 3x5x8: 0 contracts left; equals Contraction.eval
    bf16 4x5x6: 0 contracts left; equals Contraction.eval
    f32 fused: 0 contracts left; the plan fuses the sums |}]

let%expect_test "an f32 attention with both passes holds no Contract" =
  List.iter
    (fun precision ->
      let tq = 5 and tk = 7 and d = 4 and bk = 4 and mask = true in
      let dsc = E.desc ~precision ~mask ~tq ~tk ~d ~bk in
      let buffers =
        [
          E.f32 0 ~h:tq ~w:d Ssa_buffer.Input;
          E.f32 1 ~h:tk ~w:d Ssa_buffer.Input;
          E.f32 2 ~h:tk ~w:d Ssa_buffer.Input;
          E.f32 3 ~h:tq ~w:d Ssa_buffer.Output;
          E.f32 4 ~h:tq ~w:tk Ssa_buffer.Input;
        ]
      in
      let p =
        Ssa_expand_contract.program
          (Ssa_expand_attend.program (E.program ~buffers ~mask dsc))
      in
      Fmt.pr "%s: %d contracts left@."
        (A.Precision.name precision)
        (count_contracts p))
    [ A.Precision.F32_ordered; A.Precision.F32_fused ];
  [%expect
    {|
    f32_ordered: 0 contracts left
    f32_fused: 0 contracts left |}]
