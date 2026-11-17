(* A contraction called on views of its buffers, a block of rows at a time,
   computes the bits the whole-tensor [Contract] computes; so does its
   expansion into loops. A base that lets the footprint leave its buffer is
   refused by the verifier, and the operation is refused where a target has no
   form for it. *)

open Ssa_ir
module F = Ssa_fixtures
module B = Ssa_builder

let i64 = Int64.of_int

let desc ~m ~k ~n =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:1L ~m:(i64 m) ~n:(i64 n) ~k:(i64 k)
          ~a:{ Contraction.Operand.batch = 0L; outer = i64 k; k = 1L }
          ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = i64 n }
          ~c:{ Contraction.Output.batch = 0L; row = i64 n; col = 1L }
          ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
          ~numerics:Contraction.Numerics.Ordered))

let buffers ~m ~k ~n =
  [
    F.buffer 0 ~h:(i64 m) ~w:(i64 k) Ssa_format.F32 Ssa_buffer.Input;
    F.buffer 1 ~h:(i64 k) ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Input;
    F.buffer 2 ~h:(i64 m) ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Output;
  ]

let whole ~m ~k ~n =
  F.build ~buffers:(buffers ~m ~k ~n) (fun bld ->
      B.contract bld (desc ~m ~k ~n) ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2))

(* Blocks of [r] rows from a loop over [blocks] full blocks, then the tail, each
   call over views whose bases move by the block. *)
let banded_program ~m ~k ~n ~r ~blocks =
  B.program ~buffers:(buffers ~m ~k ~n) (fun bld ->
      let call bld ~rows ~at =
        let base stride = B.index_scale bld (i64 stride) at in
        B.view_call bld (desc ~m:rows ~k ~n)
          ~a:(F.buf 0, B.Base (base (r * k)))
          ~b:(F.buf 1, B.Base (B.index bld 0L))
          ~c:(F.buf 2, B.Base (base (r * n)))
      in
      ignore
        (B.for_ bld ~lo:(B.index bld 0L)
           ~hi:(B.index bld (i64 blocks))
           ~init:B.Nil
           (fun bld blk B.Nil ->
             call bld ~rows:r ~at:blk;
             B.Nil));
      let tail = m - (blocks * r) in
      if tail > 0 then call bld ~rows:tail ~at:(B.index bld (i64 blocks)))

let banded ~m ~k ~n ~r =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (banded_program ~m ~k ~n ~r ~blocks:(m / r))

let lcg = ref 11

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let run p ~a ~b ~nc =
  let memory =
    F.memory
      [ (0, F.floats a); (1, F.floats b); (2, F.floats (Array.make nc 0.)) ]
  in
  F.run p ~memory;
  match Ssa_id.Buffer.Map.find (F.buf 2) memory with
  | Ssa_memory.Floats cs -> cs
  | _ -> assert false

let same x y =
  Array.for_all2 (fun p q -> Int32.bits_of_float p = Int32.bits_of_float q) x y

let count_view_calls p =
  let n = ref 0 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr { op = Ssa_op.View_call _; _ } -> incr n
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For f -> region f.body
    | Ssa_stmt.If f ->
        region f.then_;
        region f.else_
    | Ssa_stmt.Ordered_sum f -> region f.body
  in
  region p.Ssa_program.entry;
  !n

let case ~m ~k ~n ~r =
  let a = Array.init (m * k) (fun _ -> value ())
  and b = Array.init (k * n) (fun _ -> value ()) in
  let nc = m * n in
  let want = run (whole ~m ~k ~n) ~a ~b ~nc in
  let p = banded ~m ~k ~n ~r in
  let got = run p ~a ~b ~nc in
  let expanded = Ssa_expand_view_call.program p in
  let through = run (Ssa_precision.to_f32 ~exact:true expanded) ~a ~b ~nc in
  Fmt.pr "m=%d r=%d: view calls %d; banded %s; expanded (%d left) %s@." m r
    (count_view_calls p)
    (if same want got then "equals Contract" else "DIFFERS")
    (count_view_calls expanded)
    (if same want through then "equals Contract" else "DIFFERS")

let%expect_test "banded view calls equal the whole Contract" =
  case ~m:6 ~k:4 ~n:3 ~r:2;
  case ~m:7 ~k:5 ~n:4 ~r:3;
  case ~m:5 ~k:3 ~n:2 ~r:5;
  [%expect
    {|
    m=6 r=2: view calls 1; banded equals Contract; expanded (0 left) equals Contract
    m=7 r=3: view calls 2; banded equals Contract; expanded (0 left) equals Contract
    m=5 r=5: view calls 1; banded equals Contract; expanded (0 left) equals Contract |}]

let%expect_test "a base past the last block is refused, with the unproven bound"
    =
  (match Err.payload (banded_program ~m:6 ~k:4 ~n:3 ~r:2 ~blocks:4) with
  | Ok _ -> Fmt.pr "accepted@."
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e);
  [%expect
    {| r0, stmt3: a proof of an access staying in its buffer cannot be re-derived |}]

let%expect_test "the CFG executor runs the banded program to the same bits" =
  let m = 7 and k = 5 and n = 4 and r = 3 in
  let a = Array.init (m * k) (fun _ -> value ())
  and b = Array.init (k * n) (fun _ -> value ()) in
  let want = run (whole ~m ~k ~n) ~a ~b ~nc:(m * n) in
  let cfg =
    Err.or_raise ~pp_error:Ssa_cfg_lower.pp_error
      (Ssa_cfg_lower.program (banded ~m ~k ~n ~r))
  in
  let memory =
    F.memory
      [
        (0, F.floats a); (1, F.floats b); (2, F.floats (Array.make (m * n) 0.));
      ]
  in
  Err.or_raise ~pp_error:Ssa_cfg_interp.pp_error
    (Ssa_cfg_interp.run cfg ~memory);
  (match Ssa_id.Buffer.Map.find (F.buf 2) memory with
  | Ssa_memory.Floats got ->
      Fmt.pr "%s@." (if same want got then "equals Contract" else "DIFFERS")
  | _ -> assert false);
  [%expect {| equals Contract |}]

let%expect_test "the printer names each operand's base" =
  Fmt.pr "%a@." Ssa_pp.pp (banded ~m:4 ~k:2 ~n:2 ~r:2);
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 4, 2, 1]
    buffer b1 in f32 [1, 1, 1, 2, 2, 1]
    buffer b2 out f32 [1, 1, 1, 4, 2, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:effect) = for %2 to %1 step 1 iter(%4:index, %5:effect := %0) {
        (%6:index, %7:effect) = index.scale 4, %4 effect %5
        (%8:index) = const 0:index
        (%9:index, %10:effect) = index.scale 4, %4 effect %7
        (%11:effect) = view contract f32 batch=1 m=2 n=2 k=2 a=(0,2,1) b=(0,1,2) c=(0,2,1) seed=zero ordered a=b0+%9 b=b1+%8 c=b2+%6 effect %10
        yield %11
      }
      yield %3
    } |}]

(* Rows placed by one base each, in any order: A's rows reversed, C's rows
   scattered, B's columns swapped, against the same product over arrays
   permuted by hand. *)
let rows_program ~m ~k ~n ~a_rows ~b_cols ~c_rows =
  B.program ~buffers:(buffers ~m ~k ~n) (fun bld ->
      let idx l = List.map (fun x -> B.index bld (i64 x)) l in
      B.view_call bld (desc ~m ~k ~n)
        ~a:(F.buf 0, B.Rows (idx (List.map (fun r -> r * k) a_rows)))
        ~b:(F.buf 1, B.Rows (idx b_cols))
        ~c:(F.buf 2, B.Rows (idx (List.map (fun r -> r * n) c_rows))))

let%expect_test "per-row bases place each row independently" =
  let m = 4 and k = 3 and n = 2 in
  let a = Array.init (m * k) (fun _ -> value ())
  and b = Array.init (k * n) (fun _ -> value ()) in
  let a_rows = [ 3; 1; 0; 2 ]
  and b_cols = [ 1; 0 ]
  and c_rows = [ 2; 0; 3; 1 ] in
  let p =
    Err.or_raise ~pp_error:Ssa_verify.pp_error
      (rows_program ~m ~k ~n ~a_rows ~b_cols ~c_rows)
  in
  let got = run p ~a ~b ~nc:(m * n) in
  (* output row c_rows.(i) holds row a_rows.(i) of A times B with its columns
     taken in b_cols order *)
  let a' =
    Array.init (m * k) (fun x -> a.((List.nth a_rows (x / k) * k) + (x mod k)))
  and b' =
    Array.init (k * n) (fun x -> b.((x / n * n) + List.nth b_cols (x mod n)))
  in
  let product = run (whole ~m ~k ~n) ~a:a' ~b:b' ~nc:(m * n) in
  let want = Array.make (m * n) 0. in
  List.iteri (fun i r -> Array.blit product (i * n) want (r * n) n) c_rows;
  let through =
    run
      (Ssa_precision.to_f32 ~exact:true (Ssa_expand_view_call.program p))
      ~a ~b ~nc:(m * n)
  in
  Fmt.pr "%s; expanded %s@."
    (if same want got then "rows land where their bases say" else "DIFFERS")
    (if same want through then "equal" else "DIFFERS");
  [%expect {| rows land where their bases say; expanded equal |}]

let%expect_test "a row base that leaves the buffer, or a short list, is refused"
    =
  let m = 4 and k = 3 and n = 2 in
  let show = function
    | Ok _ -> Fmt.pr "accepted@."
    | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e
  in
  show
    (Err.payload
       (rows_program ~m ~k ~n ~a_rows:[ 3; 1; 0; 4 ] ~b_cols:[ 1; 0 ]
          ~c_rows:[ 2; 0; 3; 1 ]));
  (try
     show
       (Err.payload
          (rows_program ~m ~k ~n ~a_rows:[ 3; 1; 0 ] ~b_cols:[ 1; 0 ]
             ~c_rows:[ 2; 0; 3; 1 ]))
   with Invalid_argument s -> Fmt.pr "builder: %s@." s);
  [%expect
    {|
    r0, stmt10: a proof of an access staying in its buffer cannot be re-derived
    builder: Ssa_builder: a view's row bases are not as many as the axis they place |}]

(* A ring read through the induction value: call [i] reads the [(2i + 1) mod
   slots]th slot of A, a slot being the two rows the call covers, and writes
   its own two rows of C. The modulus is the value less [modulus] times its
   floor quotient, the form the range analysis proves in [0, modulus). The
   unrolled form names each slot as a constant. *)
let ring_program ?(looped = true) ~slots ~modulus ~calls () =
  let m = 2 and k = 3 and n = 2 in
  let buffers =
    [
      F.buffer 0 ~h:(i64 (m * slots)) ~w:(i64 k) Ssa_format.F32 Ssa_buffer.Input;
      F.buffer 1 ~h:(i64 k) ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Input;
      F.buffer 2
        ~h:(i64 (m * calls))
        ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  B.program ~buffers (fun bld ->
      let call bld ~slot ~at =
        B.view_call bld (desc ~m ~k ~n)
          ~a:(F.buf 0, B.Base slot)
          ~b:(F.buf 1, B.Base (B.index bld 0L))
          ~c:(F.buf 2, B.Base (B.index_scale bld (i64 (m * n)) at))
      in
      if looped then
        ignore
          (B.for_ bld ~lo:(B.index bld 0L)
             ~hi:(B.index bld (i64 calls))
             ~init:B.Nil
             (fun bld i B.Nil ->
               let t =
                 B.index_add bld (B.index_scale bld 2L i) (B.index bld 1L)
               in
               let q = B.index_floor_div bld (i64 modulus) t in
               let r =
                 B.index_add bld t (B.index_scale bld (i64 (-modulus)) q)
               in
               call bld ~slot:(B.index_scale bld (i64 (m * k)) r) ~at:i;
               B.Nil))
      else
        for i = 0 to calls - 1 do
          call bld
            ~slot:(B.index bld (i64 (m * k * (((2 * i) + 1) mod slots))))
            ~at:(B.index bld (i64 i))
        done)

let%expect_test
    "a base modulo the ring is proved inside it, and equals the unrolled calls"
    =
  let slots = 3 and calls = 7 in
  let a = Array.init (2 * slots * 3) (fun _ -> value ())
  and b = Array.init 6 (fun _ -> value ()) in
  let nc = 2 * calls * 2 in
  let build ?looped modulus =
    Err.or_raise ~pp_error:Ssa_verify.pp_error
      (ring_program ?looped ~slots ~modulus ~calls ())
  in
  let want = run (build ~looped:false slots) ~a ~b ~nc in
  let p = build slots in
  let cfg =
    Err.or_raise ~pp_error:Ssa_cfg_lower.pp_error (Ssa_cfg_lower.program p)
  in
  let memory =
    F.memory
      [ (0, F.floats a); (1, F.floats b); (2, F.floats (Array.make nc 0.)) ]
  in
  Err.or_raise ~pp_error:Ssa_cfg_interp.pp_error
    (Ssa_cfg_interp.run cfg ~memory);
  let cfg_got =
    match Ssa_id.Buffer.Map.find (F.buf 2) memory with
    | Ssa_memory.Floats cs -> cs
    | _ -> assert false
  in
  let verdict got = if same want got then "equal" else "DIFFERS" in
  Fmt.pr "interpreter %s; CFG executor %s; expanded %s@."
    (verdict (run p ~a ~b ~nc))
    (verdict cfg_got)
    (verdict
       (run
          (Ssa_precision.to_f32 ~exact:true (Ssa_expand_view_call.program p))
          ~a ~b ~nc));
  [%expect {| interpreter equal; CFG executor equal; expanded equal |}]

let%expect_test "a modulus past the ring is refused" =
  let show modulus =
    match Err.payload (ring_program ~slots:3 ~modulus ~calls:7 ()) with
    | Ok _ -> Fmt.pr "modulus %d: accepted@." modulus
    | Error e -> Fmt.pr "modulus %d: %a@." modulus Ssa_verify.pp_error e
  in
  List.iter show [ 2; 3; 4 ];
  [%expect
    {|
    modulus 2: accepted
    modulus 3: accepted
    modulus 4: r0, stmt9: a proof of an access staying in its buffer cannot be re-derived |}]
