(* The attention operation: a program holding one [Attend] verifies, prints,
   interprets to [Attention_block.eval]'s bits, and is refused when its output
   is a buffer it reads, when a buffer's format is not the operand's, or when an
   operand reaches past its buffer. *)

open Ssa_ir
module F = Ssa_fixtures
module A = Attention_block

let strides batch head row col = { A.Strides.batch; head; row; col }

let desc ~tq ~tk ~d =
  let i = Int64.of_int in
  Result.get_ok
    (Err.payload
       (A.make ~batch:1L ~heads:1L ~tq:(i tq) ~tk:(i tk) ~d:(i d)
          ~q:(strides 0L 0L (i d) 1L)
          ~k:(strides 0L 0L (i d) 1L)
          ~v:(strides 0L 0L (i d) 1L)
          ~o:(strides 0L 0L (i d) 1L)
          ~scale:(A.Scale.default ~d:(i d))
          ~block_keys:4L ()))

(* An empty program over [buffers], its body replaced by one [Attend]. *)
let program ~buffers ~q ~k ~v ~o d =
  let p = F.build ~buffers (fun _ -> ()) in
  let effect =
    match p.Ssa_program.entry.Ssa_region.params with
    | [ e ] -> e
    | _ -> assert false
  in
  let id, next = Ssa_id.Value.Next.alloc p.Ssa_program.next_value in
  let after = { Ssa_value.id; ty = Ssa_type.Effect } in
  {
    p with
    Ssa_program.entry =
      {
        p.Ssa_program.entry with
        Ssa_region.body =
          [
            Ssa_stmt.Instr
              {
                Ssa_instr.results = [ after ];
                op =
                  Ssa_op.Attend
                    {
                      desc = d;
                      q = F.buf q;
                      k = F.buf k;
                      v = F.buf v;
                      mask = None;
                      o = F.buf o;
                    };
                token = Some effect;
                origin = Ssa_origin.Unknown;
              };
          ];
        yields = [ after ];
      };
    next_value = next;
  }

let bf16 id ~h ~w = F.buffer id ~h ~w Ssa_format.Bf16 Ssa_buffer.Input

let%expect_test "an attention verifies, interprets and is refused when wrong" =
  let tq = 5 and tk = 7 and d = 4 in
  let dsc = desc ~tq ~tk ~d in
  let i = Int64.of_int in
  let buffers =
    [
      bf16 0 ~h:(i tq) ~w:(i d);
      bf16 1 ~h:(i tk) ~w:(i d);
      bf16 2 ~h:(i tk) ~w:(i d);
      F.buffer 3 ~h:(i tq) ~w:(i d) Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  let p = program ~buffers ~q:0 ~k:1 ~v:2 ~o:3 dsc in
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> Fmt.pr "verifies@."
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e);
  Fmt.pr "%a@." Ssa_pp.pp p;
  (* bfloat16 bits of small fractions *)
  let bits x =
    Int32.to_int
      (Int32.shift_right_logical (Int32.bits_of_float (A.bf16_round x)) 16)
  in
  let cell n = bits (float_of_int ((n * 7 mod 13) - 6) /. 4.) in
  let q = Array.init (tq * d) cell
  and k = Array.init (tk * d) (fun n -> cell (n + 3))
  and v = Array.init (tk * d) (fun n -> cell (n + 9)) in
  let memory =
    F.memory
      [
        (0, Ssa_memory.Ints q);
        (1, Ssa_memory.Ints k);
        (2, Ssa_memory.Ints v);
        (3, F.floats (Array.make (tq * d) 0.));
      ]
  in
  F.run p ~memory;
  let got =
    match Ssa_id.Buffer.Map.find (F.buf 3) memory with
    | Ssa_memory.Floats a -> a
    | _ -> assert false
  in
  let want = Array.make (tq * d) 0. in
  let widen a n =
    Int32.float_of_bits (Int32.shift_left (Int32.of_int a.(Int64.to_int n)) 16)
  in
  A.eval dsc ~q:(widen q) ~k:(widen k) ~v:(widen v)
    ~mask:(fun _ -> 0.)
    ~write:(fun o x -> want.(Int64.to_int o) <- x);
  Fmt.pr "interpreter %s the reference@."
    (if
       Array.for_all2
         (fun a b -> Int32.bits_of_float a = Int32.bits_of_float b)
         got want
     then "equals"
     else "DIFFERS from");
  (* the output is a buffer it reads *)
  let aliased = program ~buffers ~q:3 ~k:1 ~v:2 ~o:3 dsc in
  (match Err.payload (Ssa_verify.check aliased) with
  | Ok () -> Fmt.pr "aliased: verifies@."
  | Error e -> Fmt.pr "aliased: %a@." Ssa_verify.pp_error e);
  (* Q declared binary32 *)
  let wrong =
    program
      ~buffers:
        (F.buffer 0 ~h:(i tq) ~w:(i d) Ssa_format.F32 Ssa_buffer.Input
        :: List.tl buffers)
      ~q:0 ~k:1 ~v:2 ~o:3 dsc
  in
  (match Err.payload (Ssa_verify.check wrong) with
  | Ok () -> Fmt.pr "f32 query: verifies@."
  | Error e -> Fmt.pr "f32 query: %a@." Ssa_verify.pp_error e);
  (* K one row short *)
  let short =
    program
      ~buffers:
        [
          List.nth buffers 0;
          bf16 1 ~h:(i (tk - 1)) ~w:(i d);
          List.nth buffers 2;
          List.nth buffers 3;
        ]
      ~q:0 ~k:1 ~v:2 ~o:3 dsc
  in
  (match Err.payload (Ssa_verify.check short) with
  | Ok () -> Fmt.pr "short key: verifies@."
  | Error e -> Fmt.pr "short key: %a@." Ssa_verify.pp_error e);
  [%expect
    {|
    verifies
    buffer b0 in bf16 [1, 1, 1, 5, 4, 1]
    buffer b1 in bf16 [1, 1, 1, 7, 4, 1]
    buffer b2 in bf16 [1, 1, 1, 7, 4, 1]
    buffer b3 out f32 [1, 1, 1, 5, 4, 1]
    entry(%0:effect) {
      (%1:effect) = attend batch=1 heads=1 tq=5 tk=7 d=4 q=(0,0,4,1) k=(0,0,4,1) v=(0,0,4,1) o=(0,0,4,1) mask=none scale=0x1p-1 block_keys=4 q=b0 k=b1 v=b2 mask=none o=b3 effect %0
      yield %1
    }

    interpreter equals the reference
    aliased: r0, stmt0: b3 is declared f32 but accessed as bf16
    f32 query: r0, stmt0: b0 is declared f32 but accessed as bf16
    short key: r0, stmt0: a proof of an access staying in its buffer cannot be re-derived |}]
