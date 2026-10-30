(* The fused attention function on the VME core, called directly: its output
   bitwise against the descriptor's streaming reference, [Attention_block.eval],
   over ragged shapes, masks, batches and heads, and each defect the suite must
   detect. *)

module A = Attention_block
module S = A.Strides
module Z = Coralnpu_kernels_zvt
module H = Kernel_harness

let strides batch head row col = { S.batch; head; row; col }

(* bfloat16 values with a spread: small fractions, now and then a large one *)
let lcg = ref 12345

let next () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  !lcg

let bf16_value () =
  let n = next () in
  Half.Bf16.of_float
    (if n mod 29 = 0 then float_of_int ((n / 29 mod 61) - 30)
     else float_of_int ((n / 7 mod 33) - 16) /. 8.)

let bf16_bytes xs =
  let b = Bytes.create (2 * Array.length xs) in
  Array.iteri (fun i x -> Bytes.set_uint16_le b (2 * i) x) xs;
  Bytes.to_string b

let bf16_of s i = Half.Bf16.to_float (String.get_uint16_le s (2 * i))

type mask = No_mask | Dense | Per_key | Per_query

let mask_value () =
  match next () mod 11 with
  | 0 | 1 -> Float.neg_infinity
  | 2 -> 3.5
  | n -> float_of_int (n - 6) /. 4.

let make ?(batch = 1) ?(heads = 1) ?(shared_kv = false) ?(mask = No_mask) ~tq
    ~tk ~d ~bk () =
  let i64 = Int64.of_int in
  let kv_head = if shared_kv then 0 else tk * d in
  let kv_batch = if shared_kv then tk * d else heads * tk * d in
  Result.get_ok
    (Err.payload
       (A.make ~batch:(i64 batch) ~heads:(i64 heads) ~tq:(i64 tq) ~tk:(i64 tk)
          ~d:(i64 d)
          ~q:(strides (i64 (heads * tq * d)) (i64 (tq * d)) (i64 d) 1L)
          ~k:(strides (i64 kv_batch) (i64 kv_head) (i64 d) 1L)
          ~v:(strides (i64 kv_batch) (i64 kv_head) (i64 d) 1L)
          ~o:(strides (i64 (heads * tq * d)) (i64 (tq * d)) (i64 d) 1L)
          ?mask:
            (match mask with
            | No_mask -> None
            | Dense -> Some (strides 0L 0L (i64 tk) 1L)
            | Per_key -> Some (strides 0L 0L 0L 1L)
            | Per_query -> Some (strides 0L 0L 1L 0L))
          ~scale:(A.Scale.default ~d:(i64 d))
          ~block_keys:(i64 bk) ()))

type outcome = {
  wrong : int;
  cells : int;
  cycles : int64;
  code : int;
  segments : (string * int) list;
}

(* [q], [k], [v] bfloat16 bits; [mask] binary32 *)
let run ?mutation ?dtcm ?(pad = 0) ?(fill = '\000') ?(sram = false) ?inputs t =
  let ( let* ) = Result.bind in
  let n x = Int64.to_int x in
  let batch = n t.A.batch and heads = n t.A.heads in
  let tq = n t.A.tq and tk = n t.A.tk and d = n t.A.d in
  let kv_heads = if t.A.k.S.head = 0L then 1 else heads in
  let q, k, v, mask =
    match inputs with
    | Some x -> x
    | None ->
        ( Array.init (batch * heads * tq * d) (fun _ -> bf16_value ()),
          Array.init (batch * kv_heads * tk * d) (fun _ -> bf16_value ()),
          Array.init (batch * kv_heads * tk * d) (fun _ -> bf16_value ()),
          match t.A.mask with
          | None -> [||]
          | Some m ->
              let rows = if m.S.row = 0L then 1 else tq
              and cols = if m.S.col = 0L then 1 else tk in
              Array.init (rows * cols) (fun _ -> mask_value ()) )
  in
  let qb = bf16_bytes q and kb = bf16_bytes k and vb = bf16_bytes v in
  let mb = H.f32_bytes mask in
  let expected =
    let out = Array.make (batch * heads * tq * d) 0. in
    A.eval t
      ~q:(fun o -> bf16_of qb (n o))
      ~k:(fun o -> bf16_of kb (n o))
      ~v:(fun o -> bf16_of vb (n o))
      ~mask:(fun o -> mask.(n o))
      ~write:(fun o x -> out.(n o) <- x);
    H.f32_bytes out
  in
  let* m = Z.Zvt_sdpa.module_ ?mutation ?dtcm t in
  let has_mask = t.A.mask <> None in
  let* r =
    H.run ~symbol:(A.symbol t) ~modules:[ m ] ~pad ~fill ~sram
      ~args:
        (if has_mask then [ `Slot 0; `Slot 1; `Slot 2; `Slot 4; `Slot 3 ]
         else [ `Slot 0; `Slot 1; `Slot 2; `Slot 3; `Imm 0 ])
      ~inputs:([ qb; kb; vb ] @ if has_mask then [ mb ] else [])
      ~outputs:[ String.length expected ]
      ()
  in
  let out = List.hd r.H.outputs in
  let cells = String.length expected / 4 in
  let wrong = ref 0 in
  for i = 0 to cells - 1 do
    if
      not
        (H.same_f32
           (Int32.float_of_bits (String.get_int32_le out (4 * i)))
           (Int32.float_of_bits (String.get_int32_le expected (4 * i))))
    then incr wrong
  done;
  (* the exact bits, a NaN's payload and a zero's sign included *)
  let exact = ref 0 in
  for i = 0 to cells - 1 do
    if String.get_int32_le out (4 * i) <> String.get_int32_le expected (4 * i)
    then incr exact
  done;
  Ok
    {
      wrong = !wrong;
      cells;
      cycles = r.H.cycles;
      code = !exact;
      segments = r.H.segments;
    }

let show ?mutation ?pad ?fill ?sram ?inputs what t =
  match run ?mutation ?pad ?fill ?sram ?inputs t with
  | Ok o ->
      Fmt.pr "%s: %s, %Ld cycles@." what
        (if o.wrong = 0 && o.code = 0 then "bitwise"
         else
           Fmt.str "DIFFERS in %d of %d cells (%d in exact bits)" o.wrong
             o.cells o.code)
        o.cycles
  | Error e -> Fmt.pr "%s: %s@." what e

let neg_inf = Float.neg_infinity
let bits x = Half.Bf16.of_float x
let zeros n = Array.make n (bits 0.)

let%expect_test "shapes: whole blocks, ragged ones, both block sizes" =
  List.iter
    (fun (tq, tk, d, bk) ->
      show (Fmt.str "%dx%d d%d bk%d" tq tk d bk) (make ~tq ~tk ~d ~bk ()))
    [
      (16, 32, 32, 32);
      (32, 32, 32, 32);
      (64, 64, 64, 32);
      (9, 29, 64, 32);
      (41, 61, 64, 32);
      (16, 36, 32, 32);
      (23, 40, 16, 16);
      (5, 7, 8, 4);
      (41, 61, 96, 16);
      (16, 32, 96, 16);
      (9, 29, 80, 16);
    ];
  [%expect
    {|
    16x32 d32 bk32: bitwise, 43943 cycles
    32x32 d32 bk32: bitwise, 87641 cycles
    64x64 d64 bk32: bitwise, 385994 cycles
    9x29 d64 bk32: bitwise, 44771 cycles
    41x61 d64 bk32: bitwise, 275669 cycles
    16x36 d32 bk32: bitwise, 52400 cycles
    23x40 d16 bk16: bitwise, 108493 cycles
    5x7 d8 bk4: bitwise, 12338 cycles
    41x61 d96 bk16: bitwise, 338555 cycles
    16x32 d96 bk16: bitwise, 64072 cycles
    9x29 d80 bk16: bitwise, 51977 cycles |}]

let%expect_test "masks: dense, per key, per query, a masked column" =
  List.iter
    (fun (tq, tk, d, bk, mask, what) ->
      show
        (Fmt.str "%dx%d d%d bk%d %s" tq tk d bk what)
        (make ~mask ~tq ~tk ~d ~bk ()))
    [
      (9, 29, 64, 32, Dense, "dense");
      (41, 61, 64, 32, Dense, "dense");
      (41, 61, 32, 32, Per_key, "[1, Tk]");
      (41, 61, 32, 32, Per_query, "[Tq, 1]");
      (41, 61, 96, 16, Dense, "dense");
    ];
  (* the second query sees nothing, the third only its last key *)
  let tq = 4 and tk = 70 and d = 32 in
  let t = make ~mask:Dense ~tq ~tk ~d ~bk:32 () in
  let mask =
    Array.init (tq * tk) (fun i ->
        let qi = i / tk and key = i mod tk in
        match qi with
        | 1 -> neg_inf
        | 2 -> if key = tk - 1 then 0. else neg_inf
        | _ -> float_of_int (key mod 5) /. 4.)
  in
  show "masked columns"
    ~inputs:
      ( Array.init (tq * d) (fun _ -> bf16_value ()),
        Array.init (tk * d) (fun _ -> bf16_value ()),
        Array.init (tk * d) (fun _ -> bf16_value ()),
        mask )
    t;
  [%expect
    {|
    9x29 d64 bk32 dense: bitwise, 45219 cycles
    41x61 d64 bk32 dense: bitwise, 286030 cycles
    41x61 d32 bk32 [1, Tk]: bitwise, 243920 cycles
    41x61 d32 bk32 [Tq, 1]: bitwise, 241930 cycles
    41x61 d96 bk16 dense: bitwise, 349278 cycles
    masked columns: bitwise, 92357 cycles |}]

let%expect_test "batches and heads, K and V shared by the heads" =
  show "2 batches x 3 heads, K/V per batch"
    (make ~batch:2 ~heads:3 ~shared_kv:true ~tq:9 ~tk:29 ~d:32 ~bk:32 ());
  show "2 batches x 3 heads, masked, distinct K/V"
    (make ~batch:2 ~heads:3 ~mask:Per_key ~tq:9 ~tk:29 ~d:32 ~bk:16 ());
  [%expect
    {|
    2 batches x 3 heads, K/V per batch: bitwise, 235080 cycles
    2 batches x 3 heads, masked, distinct K/V: bitwise, 253315 cycles |}]

(* The key block is part of the grant: zero Q and K, 32 keys, d = 64, mask 0 on
   the first 16 keys and 1 on the last 16, V = 1 on the first 16 keys. *)
let%expect_test "the key block decides the answer" =
  let tq = 1 and tk = 32 and d = 64 in
  let inputs =
    ( zeros (tq * d),
      zeros (tk * d),
      Array.init (tk * d) (fun i -> bits (if i / d < 16 then 1. else 0.)),
      Array.init tk (fun r -> if r < 16 then 0. else 1.) )
  in
  List.iter
    (fun bk ->
      show (Fmt.str "bk %d" bk) ~inputs (make ~mask:Per_key ~tq ~tk ~d ~bk ()))
    [ 32; 16 ];
  [%expect
    {|
    bk 32: bitwise, 45028 cycles
    bk 16: bitwise, 47241 cycles |}]

(* V is staged, never read in place: a V that ends where the neighbour holds
   Inf or NaN bytes, in SRAM and in the DTCM, ragged and with a full last block *)
let%expect_test "V's neighbours" =
  List.iter
    (fun (sram, tk) ->
      List.iter
        (fun fill ->
          show ~pad:1024 ~fill ~sram
            (Fmt.str "tk %d, %s, after-bytes %02x" tk
               (if sram then "SRAM" else "DTCM")
               (Char.code fill))
            (make ~tq:16 ~tk ~d:64 ~bk:32 ()))
        [ '\xFF'; '\x7F' ])
    [ (true, 29); (false, 29); (true, 64); (false, 64) ];
  [%expect
    {|
    tk 29, SRAM, after-bytes ff: bitwise, 48687 cycles
    tk 29, SRAM, after-bytes 7f: bitwise, 48511 cycles
    tk 29, DTCM, after-bytes ff: bitwise, 48163 cycles
    tk 29, DTCM, after-bytes 7f: bitwise, 48236 cycles
    tk 64, SRAM, after-bytes ff: bitwise, 96735 cycles
    tk 64, SRAM, after-bytes 7f: bitwise, 96744 cycles
    tk 64, DTCM, after-bytes ff: bitwise, 96418 cycles
    tk 64, DTCM, after-bytes 7f: bitwise, 96406 cycles |}]

(* Tq 16, Tk 34, d 64, zero Q and K, mask 0 on the first 32 keys and 200 on the
   last two, V feature 0 -1 then -0: the engine's zero sum is +0. *)
let%expect_test "a negative-zero seed meets a short last block" =
  let tq = 16 and tk = 34 and d = 64 in
  let t = make ~mask:Per_key ~tq ~tk ~d ~bk:32 () in
  let inputs =
    ( zeros (tq * d),
      zeros (tk * d),
      Array.init (tk * d) (fun i ->
          if i mod d = 0 then if i / d < 32 then bits (-1.) else 0x8000
          else bits 1.),
      Array.init tk (fun r -> if r < 32 then 0. else 200.) )
  in
  show "tk 34" ~inputs t;
  [%expect {| tk 34: bitwise, 60283 cycles |}]

(* The function's size, the largest tile function's and the image's, against the
   instruction memory's 8 KiB *)
let%expect_test "code size" =
  let t = make ~mask:Dense ~tq:41 ~tk:61 ~d:64 ~bk:32 () in
  (match run t with
  | Ok o ->
      List.iter (fun (n, size) -> Fmt.pr "%s: %d bytes@." n size) o.segments
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {|
    .text: 6128 bytes
    .data: 20 bytes
    .bss: 17664 bytes |}]

let%expect_test "code size at d 96" =
  let t = make ~mask:Dense ~tq:41 ~tk:61 ~d:96 ~bk:16 () in
  (match run t with
  | Ok o ->
      List.iter (fun (n, size) -> Fmt.pr "%s: %d bytes@." n size) o.segments
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {|
    .text: 6620 bytes
    .data: 20 bytes
    .bss: 17920 bytes |}]

let%expect_test "what d 96 and 130 need" =
  List.iter
    (fun (d, bk) ->
      Fmt.pr "d %d bk %d: %a@." d bk
        Fmt.(result ~ok:int ~error:string)
        (Z.Zvt_sdpa.panel_bytes (make ~tq:16 ~tk:32 ~d ~bk ())))
    [ (96, 32); (96, 16); (130, 16) ];
  [%expect
    {|
    d 96 bk 32: 25088
    d 96 bk 16: 17920
    d 130 bk 16: d past 128 |}]

let dtcm =
  let profile =
    Result.get_ok (Err.payload (Coral_profile.load ~dir:H.dir H.configuration))
  in
  let r =
    List.find
      (fun (r : Coral_profile.Region.t) ->
        r.Coral_profile.Region.role = Coral_profile.Role.Dtcm)
      profile.Coral_profile.regions
  in
  ( Int64.to_int r.Coral_profile.Region.base,
    Int64.to_int r.Coral_profile.Region.size )

(* Each defect must turn a case red; the unmutated run of the same case is
   bitwise. *)
let%expect_test "each defect is detected" =
  let verdict what ?pad ?fill ?inputs ?sram ?dtcm mutation t =
    match run ~mutation ?dtcm ?pad ?fill ?inputs ?sram t with
    | Ok o ->
        Fmt.pr "%s: %s@." what
          (if o.wrong = 0 && o.code = 0 then "STILL BITWISE (vacuous)"
           else "differs")
    | Error e -> Fmt.pr "%s: %s@." what e
  in
  let multi = make ~mask:Dense ~tq:20 ~tk:61 ~d:32 ~bk:32 () in
  let module M = Z.Zvt_sdpa.Mutation in
  verdict "max stale" M.Max_stale multi;
  verdict "rescale omitted" M.Rescale_omitted multi;
  verdict "seed ignored" M.Seed_ignored multi;
  verdict "V not advanced" M.V_not_advanced multi;
  (* V's previous block ends in Inf; a short block must not see it as its tail *)
  let tk = 61 and d = 64 and tq = 16 in
  let t = make ~tq ~tk ~d ~bk:32 () in
  let inputs =
    ( Array.init (tq * d) (fun _ -> bf16_value ()),
      Array.init (tk * d) (fun _ -> bf16_value ()),
      Array.init (tk * d) (fun i ->
          let key = i / d in
          if key >= 29 && key < 32 then 0x7F80 else bf16_value ()),
      [||] )
  in
  show "V tail, unmutated" ~inputs t;
  verdict "V tail unzeroed" ~inputs M.V_tail_unzeroed t;
  (* the same at d 96, whose rows the tail clears in two vectors *)
  let tk = 29 and d = 96 and tq = 16 in
  let t = make ~tq ~tk ~d ~bk:16 () in
  let inputs =
    ( Array.init (tq * d) (fun _ -> bf16_value ()),
      Array.init (tk * d) (fun _ -> bf16_value ()),
      Array.init (tk * d) (fun i ->
          let key = i / d in
          if key >= 13 && key < 16 then 0x7F80 else bf16_value ()),
      [||] )
  in
  show "V tail d96, unmutated" ~inputs t;
  verdict "V tail d96 unzeroed" ~inputs M.V_tail_unzeroed t;
  (* V in the DTCM, ragged, with NaN bytes after it *)
  let t = make ~tq:16 ~tk:29 ~d:16 ~bk:32 () in
  show "V in the DTCM, unmutated" ~pad:1024 ~fill:'\xFF' t;
  verdict "V read in place in the DTCM" ~pad:1024 ~fill:'\xFF' ~dtcm
    M.V_in_place_in_dtcm t;
  [%expect
    {|
    max stale: differs
    rescale omitted: differs
    seed ignored: differs
    V not advanced: differs
    V tail, unmutated: bitwise, 91468 cycles
    V tail unzeroed: differs
    V tail d96, unmutated: bitwise, 58220 cycles
    V tail d96 unzeroed: differs
    V in the DTCM, unmutated: bitwise, 38077 cycles
    V read in place in the DTCM: differs |}]

(* Cycles by phase at 64x64 d64: the whole, then each phase left out in turn
   (the result is wrong; the difference is the phase's share). *)
let%expect_test "cycles by phase" =
  let t = make ~tq:64 ~tk:64 ~d:64 ~bk:32 () in
  let cycles ?skip () =
    let ( let* ) = Result.bind in
    let* m = Z.Zvt_sdpa.module_ ?skip t in
    let n = 64 * 64 in
    let q = bf16_bytes (Array.init n (fun _ -> bf16_value ()))
    and k = bf16_bytes (Array.init n (fun _ -> bf16_value ()))
    and v = bf16_bytes (Array.init n (fun _ -> bf16_value ())) in
    let* r =
      H.run ~symbol:(A.symbol t) ~modules:[ m ] ~inputs:[ q; k; v ]
        ~args:[ `Slot 0; `Slot 1; `Slot 2; `Slot 3; `Imm 0 ]
        ~outputs:[ 4 * n ]
        ()
    in
    Ok r.H.cycles
  in
  let whole = Result.get_ok (cycles ()) in
  Fmt.pr "whole: %Ld@." whole;
  List.iter
    (fun (what, phase) ->
      let c = Result.get_ok (cycles ~skip:[ phase ] ()) in
      Fmt.pr "%s: %Ld@." what (Int64.sub whole c))
    Z.Zvt_sdpa.Phase.
      [
        ("scores block", Scores);
        ("exp pass", Exp);
        ("context block", Context);
        ("normalize", Normalize);
      ];
  [%expect
    {|
    whole: 386083
    scores block: 31766
    exp pass: 290730
    context block: 29314
    normalize: 17694 |}]

(* d 128 fits the instruction memory only because the exp is one shared
   routine, called from both places that need it *)
let%expect_test "d 128" =
  List.iter
    (fun (d, bk) ->
      Fmt.pr "d %d bk %d: %a@." d bk
        Fmt.(result ~ok:int ~error:string)
        (Z.Zvt_sdpa.panel_bytes (make ~tq:16 ~tk:32 ~d ~bk ())))
    [ (128, 16); (128, 8) ];
  let t = make ~mask:Dense ~tq:41 ~tk:61 ~d:128 ~bk:16 () in
  (match run t with
  | Ok o ->
      List.iter (fun (n, size) -> Fmt.pr "%s: %d bytes@." n size) o.segments
  | Error e -> Fmt.pr "%s@." e);
  show "41x61 d128 bk16 dense" t;
  show "16x32 d128 bk16" (make ~tq:16 ~tk:32 ~d:128 ~bk:16 ());
  show "9x29 d112 bk16" (make ~tq:9 ~tk:29 ~d:112 ~bk:16 ());
  [%expect
    {|
    d 128 bk 16: 23296
    d 128 bk 8: 21632
    .text: 7036 bytes
    .data: 20 bytes
    .bss: 23296 bytes
    41x61 d128 bk16 dense: bitwise, 386836 cycles
    16x32 d128 bk16: bitwise, 72966 cycles
    9x29 d112 bk16: bitwise, 59535 cycles |}]
