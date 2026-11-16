(* A view call lowers to contraction helper calls on offset addresses (one call
   per row where bases are given per row). The lowered program, on the generic,
   AArch64 and RV32 interpreters, computes the bits of the SSA interpreter,
   which the banded-versus-whole tests tie to the whole-tensor Contract chain. *)

open Ssa_ir
open Machine_ir
open Machine_interp
module Src = Machine_source_test.Mir_source
module F = Ssa_ir_test.Ssa_fixtures
module B = Ssa_builder
module S = Band_schedule
module D = Band_driver
module A = Machine_target_aarch64
module R = Machine_target_riscv32

let i64 = Int64.of_int
let lcg = ref 5

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let ssa_output p ~inputs =
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let id = b.Ssa_buffer.id in
        let cells =
          match List.assoc_opt (Ssa_id.Buffer.to_int id) inputs with
          | Some c -> c
          | None ->
              Array.make
                (Int64.to_int
                   (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents)))
                0.
        in
        Ssa_id.Buffer.Map.add id (Ssa_memory.Floats cells) m)
      Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
  in
  Err.or_raise ~pp_error:Ssa_interp.pp_error (Ssa_interp.run p ~memory);
  List.filter_map
    (fun (b : Ssa_buffer.t) ->
      match
        (b.Ssa_buffer.role, Ssa_id.Buffer.Map.find b.Ssa_buffer.id memory)
      with
      | Ssa_buffer.Output, Ssa_memory.Floats cs ->
          Some
            (Array.map
               (fun x ->
                 Int64.logand
                   (Int64.of_int32 (Int32.bits_of_float x))
                   0xFFFF_FFFFL)
               cs)
      | _ -> None)
    p.Ssa_program.buffers

let lower ?mutation p =
  let planning =
    Mir_planning.make
      ~subject:(Machine_lower.Mir_lower.subject p)
      ~policy:"reference_f64" ~schedule:"scalar"
      ~precision:Mir_planning.Precision.F32 ~lanes:(Mir_type.Lanes.of_int 1)
      ~fma:Mir_planning.Fma.Forbidden ~capabilities:[]
  in
  Err.payload
    (Machine_lower.Mir_lower.program ?mutation ~planning:(Some planning) p)

let observed ~layout memory binding =
  List.map
    (fun (o : Mir_observation.Output.t) ->
      Array.map
        (function
          | Some (c : Mir_const.t) -> Int64.logand c.Mir_const.bits 0xFFFF_FFFFL
          | None -> -1L)
        o.Mir_observation.Output.cells)
    (Src.mir_outputs memory binding layout)

let bound ~(lowered : Machine_lower.Mir_lower.result) ~inputs =
  Src.bound_of lowered ~input:(fun b ->
      Option.map
        (fun c -> Src.pack_cells b (Ssa_memory.Floats c))
        (List.assoc_opt (Ssa_id.Buffer.to_int b.Ssa_buffer.id) inputs))

let models_of g =
  let hs = (Mir_verify.Generic.program g).Mir_program.helpers in
  Mir_math_model.all @ Mir_contract_model.models hs

let finished (o : Mir_interp.Outcome.t) =
  match o with
  | Mir_interp.Outcome.Success _ -> Ok ()
  | o -> Error (Fmt.str "%a" Mir_interp.Outcome.pp o)

let generic_run (lowered : Machine_lower.Mir_lower.result) ~inputs =
  let program = lowered.Machine_lower.Mir_lower.program in
  let memory = Mir_memory.create () in
  let binding =
    Result.get_ok
      (Mir_interp.instantiate
         (Mir_verify.Generic.program program)
         memory ~bound:(bound ~lowered ~inputs))
  in
  let r =
    Mir_interp.run ~models:(models_of program) program memory binding ~args:[]
  in
  Result.map
    (fun () ->
      observed ~layout:lowered.Machine_lower.Mir_lower.layout memory binding)
    (finished r.Mir_interp.outcome)

let a64_run (lowered : Machine_lower.Mir_lower.result) ~inputs =
  match
    Err.payload (A.A64_select.program lowered.Machine_lower.Mir_lower.program)
  with
  | Error r -> Error (Fmt.str "%a" A.A64_select.Refusal.pp r)
  | Ok res ->
      let sel = A.A64_stage.Sel.Verified.selected res.A.A64_select.selected in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.A.A64_stage.Sel.program memory
             ~bound:(bound ~lowered ~inputs))
      in
      let r =
        A.A64_stage.Interp.run
          ~models:(models_of lowered.Machine_lower.Mir_lower.program)
          res.A.A64_select.selected memory binding ~args:[]
      in
      Result.map
        (fun () ->
          observed ~layout:lowered.Machine_lower.Mir_lower.layout memory binding)
        (finished r.A.A64_stage.Interp.outcome)

let rv32_run (lowered : Machine_lower.Mir_lower.result) ~inputs =
  match
    Err.payload (R.Rv32_select.program lowered.Machine_lower.Mir_lower.program)
  with
  | Error r -> Error (Fmt.str "%a" R.Rv32_select.Refusal.pp r)
  | Ok res ->
      let sel = R.Rv32_stage.Sel.Verified.selected res.R.Rv32_select.selected in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.R.Rv32_stage.Sel.program memory
             ~bound:(bound ~lowered ~inputs))
      in
      let r =
        R.Rv32_stage.Interp.run
          ~models:(models_of lowered.Machine_lower.Mir_lower.program)
          res.R.Rv32_select.selected memory binding ~args:[]
      in
      Result.map
        (fun () ->
          observed ~layout:lowered.Machine_lower.Mir_lower.layout memory binding)
        (finished r.R.Rv32_stage.Interp.outcome)

(* [want] is the reference output; every route must equal it. *)
let verdict ?mutation ~name p ~inputs ~want =
  match lower ?mutation p with
  | Error r ->
      Fmt.pr "%s: refused: %a@." name Machine_lower.Mir_lower.Refusal.pp r
  | Ok lowered ->
      let route (rname, run) =
        match
          try run lowered ~inputs with Invalid_argument _ -> Error "defect"
        with
        | Ok got -> if got = want then rname ^ " equal" else rname ^ " DIFFERS"
        | Error e -> rname ^ " failed: " ^ e
      in
      Fmt.pr "%s: %s@." name
        (String.concat "; "
           (List.map route
              [
                ("generic", generic_run);
                ("aarch64", a64_run);
                ("rv32", rv32_run);
              ]))

(* Banded pointwise chains from the driver against the whole Contract chain. *)
let kernel name ~chunk =
  {
    S.name;
    extent = S.Count.of_int 1;
    stride = S.Count.of_int 1;
    pad = S.Count.of_int 0;
    dilation = S.Count.of_int 1;
    chunk = S.Span.chunk chunk;
  }

let stage name ~chunk ~cin ~cout =
  { D.Stage.kernel = kernel name ~chunk; cin = i64 cin; cout = i64 cout }

let elements (p : D.Program.t) id =
  Int64.to_int
    (Option.get
       (Ssa_buffer.elements
          (Option.get (Ssa_program.find_buffer p.D.Program.program id))
            .Ssa_buffer.extents))

let band ?mutation ?driver ~name stages ~width ~rows ~block =
  match
    D.program ?mutation:driver stages ~width ~rows:(S.Count.of_int rows)
      ~block:(S.Span.block block)
  with
  | Error _ -> Fmt.pr "%s: driver refused@." name
  | Ok banded ->
      let whole = D.whole stages ~width ~rows:(S.Count.of_int rows) in
      let cells id = Array.init (elements whole id) (fun _ -> value ()) in
      let inputs =
        List.map
          (fun id -> (Ssa_id.Buffer.to_int id, cells id))
          (whole.D.Program.input :: whole.D.Program.weights)
      in
      let want = ssa_output whole.D.Program.program ~inputs in
      Fmt.pr "%s: SSA banded %s; " name
        (if ssa_output banded.D.Program.program ~inputs = want then "equal"
         else "DIFFERS");
      verdict ?mutation ~name:"lowered" banded.D.Program.program ~inputs ~want

let three =
  [
    stage "a" ~chunk:2 ~cin:2 ~cout:3;
    stage "b" ~chunk:2 ~cin:3 ~cout:4;
    stage "c" ~chunk:2 ~cin:4 ~cout:2;
  ]

let desc ~batch ~m ~k ~n =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:(i64 batch) ~m:(i64 m) ~n:(i64 n) ~k:(i64 k)
          ~a:{ Contraction.Operand.batch = i64 (m * k); outer = i64 k; k = 1L }
          ~b:{ Contraction.Operand.batch = i64 (k * n); outer = 1L; k = i64 n }
          ~c:{ Contraction.Output.batch = i64 (m * n); row = i64 n; col = 1L }
          ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
          ~numerics:Contraction.Numerics.Ordered))

(* Three buffers with room for two extra rows or columns each; the views name
   where in them the descriptor's cells start. *)
let rows_program ~batch ~m ~k ~n ~a_rows ~b_cols ~c_rows =
  let slack = 2 in
  let buffers =
    [
      F.buffer 0
        ~h:(i64 (batch * (m + slack)))
        ~w:(i64 k) Ssa_format.F32 Ssa_buffer.Input;
      F.buffer 1 ~h:(i64 batch)
        ~w:(i64 (k * (n + slack)))
        Ssa_format.F32 Ssa_buffer.Input;
      F.buffer 2
        ~h:(i64 (batch * (m + slack)))
        ~w:(i64 (n + slack))
        Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers (fun bld ->
         let idx xs = List.map (fun x -> B.index bld (i64 x)) xs in
         let at buf = function
           | `Base x -> (F.buf buf, B.Base (B.index bld (i64 x)))
           | `Rows xs -> (F.buf buf, B.Rows (idx xs))
         in
         B.view_call bld (desc ~batch ~m ~k ~n) ~a:(at 0 a_rows)
           ~b:(at 1 b_cols) ~c:(at 2 c_rows)))

let rows_case ?mutation ~name ~batch ~m ~k ~n ~a_rows ~b_cols ~c_rows () =
  let p = rows_program ~batch ~m ~k ~n ~a_rows ~b_cols ~c_rows in
  let inputs =
    List.map
      (fun (b : Ssa_buffer.t) ->
        ( Ssa_id.Buffer.to_int b.Ssa_buffer.id,
          Array.init
            (Int64.to_int
               (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents)))
            (fun _ -> value ()) ))
      (List.filter
         (fun (b : Ssa_buffer.t) -> b.Ssa_buffer.role = Ssa_buffer.Input)
         p.Ssa_program.buffers)
  in
  verdict ?mutation ~name p ~inputs ~want:(ssa_output p ~inputs)

let bands ?mutation () =
  band ?mutation ~name:"3 stages, 8 rows, block 2" three ~width:3L ~rows:8
    ~block:2;
  band ?mutation ~name:"3 stages, 7 rows, block 2" three ~width:2L ~rows:7
    ~block:2;
  band ?mutation ~name:"3 stages, 8 rows, block 3" three ~width:3L ~rows:8
    ~block:3;
  (* blocks that repeat run as a loop over one body *)
  band ?mutation ~name:"3 stages, 16 rows, block 2, looped" three ~width:3L
    ~rows:16 ~block:2;
  band ?mutation ~name:"3 stages, 12 rows, block 1, looped" three ~width:2L
    ~rows:12 ~block:1;
  band ?mutation ~name:"3 stages, 48 rows, block 3, looped" three ~width:2L
    ~rows:48 ~block:3

let rows ?mutation () =
  rows_case ?mutation ~name:"A and C rows reversed" ~batch:1 ~m:3 ~k:4 ~n:3
    ~a_rows:(`Rows [ 8; 4; 0 ])
    ~b_cols:(`Base 0)
    ~c_rows:(`Rows [ 10; 5; 0 ])
    ();
  rows_case ?mutation ~name:"B columns permuted" ~batch:1 ~m:2 ~k:3 ~n:3
    ~a_rows:(`Base 3)
    ~b_cols:(`Rows [ 2; 0; 1 ])
    ~c_rows:(`Base 1) ();
  rows_case ?mutation ~name:"A single base, C rows" ~batch:1 ~m:3 ~k:2 ~n:2
    ~a_rows:(`Base 2) ~b_cols:(`Base 1)
    ~c_rows:(`Rows [ 7; 0; 3 ])
    ();
  rows_case ?mutation ~name:"batched, all three per row" ~batch:2 ~m:2 ~k:3 ~n:2
    ~a_rows:(`Rows [ 5; 1 ])
    ~b_cols:(`Rows [ 1; 0 ])
    ~c_rows:(`Rows [ 4; 0 ])
    ()

let%expect_test
    "banded chains lowered through the helper route equal the whole chain" =
  bands ();
  [%expect
    {|
    3 stages, 8 rows, block 2: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 7 rows, block 2: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 8 rows, block 3: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 16 rows, block 2, looped: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 12 rows, block 1, looped: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 48 rows, block 3, looped: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal |}]

let%expect_test "per-row bases lowered through the helper route" =
  rows ();
  [%expect
    {|
    A and C rows reversed: generic equal; aarch64 equal; rv32 equal
    B columns permuted: generic equal; aarch64 equal; rv32 equal
    A single base, C rows: generic equal; aarch64 equal; rv32 equal
    batched, all three per row: generic equal; aarch64 equal; rv32 equal |}]

(* Each lowering defect is seen: the bases dropped moves every cell, the split
   axis not moving a single base repeats the first row. *)
let%expect_test "view-call lowering mutations are red" =
  List.iter
    (fun (name, mutation) ->
      Fmt.pr "== %s@." name;
      bands ~mutation ();
      rows ~mutation ())
    Machine_lower.Mir_lower.Mutation.
      [
        ("base dropped", View_base_dropped);
        ("outer dropped", View_outer_dropped);
      ];
  [%expect
    {|
    == base dropped
    3 stages, 8 rows, block 2: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    3 stages, 7 rows, block 2: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    3 stages, 8 rows, block 3: SSA banded equal; lowered: generic failed: defect; aarch64 failed: defect; rv32 failed: defect
    3 stages, 16 rows, block 2, looped: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    3 stages, 12 rows, block 1, looped: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    3 stages, 48 rows, block 3, looped: SSA banded equal; lowered: generic failed: defect; aarch64 failed: defect; rv32 failed: defect
    A and C rows reversed: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    B columns permuted: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    A single base, C rows: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    batched, all three per row: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    == outer dropped
    3 stages, 8 rows, block 2: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 7 rows, block 2: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 8 rows, block 3: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    3 stages, 16 rows, block 2, looped: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 12 rows, block 1, looped: SSA banded equal; lowered: generic equal; aarch64 equal; rv32 equal
    3 stages, 48 rows, block 3, looped: SSA banded equal; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    A and C rows reversed: generic equal; aarch64 equal; rv32 equal
    B columns permuted: generic equal; aarch64 equal; rv32 equal
    A single base, C rows: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    batched, all three per row: generic equal; aarch64 equal; rv32 equal |}]

(* The loop's own defects: bases that do not move, a loop one repetition
   short. The banded SSA program already differs from the whole chain; the
   lowered routes run the same wrong program. *)
let%expect_test "loop mutations are red on every route" =
  List.iter
    (fun (name, driver) ->
      Fmt.pr "== %s@." name;
      band ~driver ~name:"3 stages, 16 rows, block 2, looped" three ~width:3L
        ~rows:16 ~block:2)
    D.[ ("iterations short", Iterations_short); ("step dropped", Step_dropped) ];
  [%expect
    {|
    == iterations short
    3 stages, 16 rows, block 2, looped: SSA banded DIFFERS; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS
    == step dropped
    3 stages, 16 rows, block 2, looped: SSA banded DIFFERS; lowered: generic DIFFERS; aarch64 DIFFERS; rv32 DIFFERS |}]
