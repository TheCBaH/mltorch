(* A binary32 attention expanded into loops and [Contract]s computes, once
   narrowed to binary32 and interpreted, the bits [Attention_block.eval] gives.
   A bfloat16 attention is left as it is. *)

open Ssa_ir
module F = Ssa_fixtures
module A = Attention_block

let i64 = Int64.of_int
let strides batch head row col = { A.Strides.batch; head; row; col }

(* A dense single-head attention: Q [tq][d], K and V [tk][d], mask [tq][tk]. *)
let desc ~precision ~mask ~tq ~tk ~d ~bk =
  Result.get_ok
    (Err.payload
       (A.make ~batch:1L ~heads:1L ~tq:(i64 tq) ~tk:(i64 tk) ~d:(i64 d)
          ~q:(strides 0L 0L (i64 d) 1L)
          ~k:(strides 0L 0L (i64 d) 1L)
          ~v:(strides 0L 0L (i64 d) 1L)
          ~o:(strides 0L 0L (i64 d) 1L)
          ?mask:(if mask then Some (strides 0L 0L (i64 tk) 1L) else None)
          ~scale:(A.Scale.default ~d:(i64 d))
          ~block_keys:(i64 bk) ~precision ()))

let f32 id ~h ~w role = F.buffer id ~h:(i64 h) ~w:(i64 w) Ssa_format.F32 role

(* An empty program over [buffers], its body replaced by one [Attend]. *)
let program ~buffers ~mask d =
  let p = F.build ~buffers (fun _ -> ()) in
  let effect = List.hd p.Ssa_program.entry.Ssa_region.params in
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
                      q = F.buf 0;
                      k = F.buf 1;
                      v = F.buf 2;
                      mask = (if mask then Some (F.buf 4) else None);
                      o = F.buf 3;
                    };
                token = Some effect;
                origin = Ssa_origin.Unknown;
              };
          ];
        yields = [ after ];
      };
    next_value = next;
  }

let lcg = ref 4242

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let mask_value () =
  match !lcg mod 11 with
  | 0 | 1 -> Float.neg_infinity
  | n -> float_of_int (n - 6) /. 4.

let count_ops p =
  let attend = ref 0 and contract = ref 0 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr { op = Ssa_op.Attend _; _ } -> incr attend
    | Ssa_stmt.Instr { op = Ssa_op.Contract _; _ } -> incr contract
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For f -> region f.body
    | Ssa_stmt.If f ->
        region f.then_;
        region f.else_
    | Ssa_stmt.Ordered_sum f -> region f.body
  in
  region p.Ssa_program.entry;
  (!attend, !contract)

let case ~precision ~mask ~tq ~tk ~d ~bk =
  let dsc = desc ~precision ~mask ~tq ~tk ~d ~bk in
  let buffers =
    [
      f32 0 ~h:tq ~w:d Ssa_buffer.Input;
      f32 1 ~h:tk ~w:d Ssa_buffer.Input;
      f32 2 ~h:tk ~w:d Ssa_buffer.Input;
      f32 3 ~h:tq ~w:d Ssa_buffer.Output;
    ]
    @ if mask then [ f32 4 ~h:tq ~w:tk Ssa_buffer.Input ] else []
  in
  let p = program ~buffers ~mask dsc in
  let expanded = Ssa_expand_attend.program p in
  let verifies q =
    match Err.payload (Ssa_verify.check q) with
    | Ok () -> "verifies"
    | Error e -> Fmt.str "%a" Ssa_verify.pp_error e
  in
  let attends, contracts = count_ops expanded in
  let q = Array.init (tq * d) (fun _ -> value ())
  and k = Array.init (tk * d) (fun _ -> value ())
  and v = Array.init (tk * d) (fun _ -> value ())
  and m = Array.init (tq * tk) (fun _ -> mask_value ()) in
  let narrowed = Ssa_precision.to_f32 ~exact:true expanded in
  let memory =
    F.memory
      ([
         (0, F.floats q);
         (1, F.floats k);
         (2, F.floats v);
         (3, F.floats (Array.make (tq * d) 0.));
       ]
      @ if mask then [ (4, F.floats m) ] else [])
  in
  (* the program declares its scratch; the caller binds it *)
  let memory =
    List.fold_left
      (fun memory (b : Ssa_buffer.t) ->
        match b.Ssa_buffer.role with
        | Ssa_buffer.Scratch ->
            let n = Int64.to_int (Option.get (Ssa_buffer.elements b.extents)) in
            Ssa_id.Buffer.Map.add b.Ssa_buffer.id
              (F.floats (Array.make n 0.))
              memory
        | Ssa_buffer.Input | Ssa_buffer.Output -> memory)
      memory narrowed.Ssa_program.buffers
  in
  F.run narrowed ~memory;
  let got =
    match Ssa_id.Buffer.Map.find (F.buf 3) memory with
    | Ssa_memory.Floats a -> a
    | _ -> assert false
  in
  let want = Array.make (tq * d) 0. in
  let at a o = a.(Int64.to_int o) in
  A.eval dsc ~q:(at q) ~k:(at k) ~v:(at v) ~mask:(at m) ~write:(fun o x ->
      want.(Int64.to_int o) <- x);
  Fmt.pr "%s %dx%d d%d bk%d%s: %s; %d attends, %d contracts; %s@."
    (A.Precision.name precision)
    tq tk d bk
    (if mask then " masked" else "")
    (verifies expanded) attends contracts
    (if
       Array.for_all2
         (fun a b -> Int32.bits_of_float a = Int32.bits_of_float b)
         got want
     then "equals the reference"
     else "DIFFERS from the reference")

let%expect_test "an expanded binary32 attention equals the reference" =
  List.iter
    (fun precision ->
      List.iter
        (fun (tq, tk, d, bk, mask) -> case ~precision ~mask ~tq ~tk ~d ~bk)
        [
          (5, 7, 4, 4, false);
          (5, 7, 4, 4, true);
          (3, 8, 6, 4, true);
          (4, 3, 2, 8, false);
          (9, 29, 8, 8, true);
        ])
    [ A.Precision.F32_ordered; A.Precision.F32_fused ];
  [%expect
    {|
    f32_ordered 5x7 d4 bk4: verifies; 0 attends, 4 contracts; equals the reference
    f32_ordered 5x7 d4 bk4 masked: verifies; 0 attends, 4 contracts; equals the reference
    f32_ordered 3x8 d6 bk4 masked: verifies; 0 attends, 2 contracts; equals the reference
    f32_ordered 4x3 d2 bk8: verifies; 0 attends, 2 contracts; equals the reference
    f32_ordered 9x29 d8 bk8 masked: verifies; 0 attends, 4 contracts; equals the reference
    f32_fused 5x7 d4 bk4: verifies; 0 attends, 4 contracts; equals the reference
    f32_fused 5x7 d4 bk4 masked: verifies; 0 attends, 4 contracts; equals the reference
    f32_fused 3x8 d6 bk4 masked: verifies; 0 attends, 2 contracts; equals the reference
    f32_fused 4x3 d2 bk8: verifies; 0 attends, 2 contracts; equals the reference
    f32_fused 9x29 d8 bk8 masked: verifies; 0 attends, 4 contracts; equals the reference |}]

let%expect_test "a bfloat16 attention is left alone" =
  let dsc =
    desc ~precision:A.Precision.Bf16_paired ~mask:false ~tq:5 ~tk:7 ~d:4 ~bk:4
  in
  let p =
    program
      ~buffers:
        [
          F.buffer 0 ~h:5L ~w:4L Ssa_format.Bf16 Ssa_buffer.Input;
          F.buffer 1 ~h:7L ~w:4L Ssa_format.Bf16 Ssa_buffer.Input;
          F.buffer 2 ~h:7L ~w:4L Ssa_format.Bf16 Ssa_buffer.Input;
          F.buffer 3 ~h:5L ~w:4L Ssa_format.F32 Ssa_buffer.Output;
        ]
      ~mask:false dsc
  in
  let q = Ssa_expand_attend.program p in
  let attends, contracts = count_ops q in
  Fmt.pr "%d attends, %d contracts, %d buffers@." attends contracts
    (List.length q.Ssa_program.buffers);
  [%expect {| 1 attends, 0 contracts, 4 buffers |}]
