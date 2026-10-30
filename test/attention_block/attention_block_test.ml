(* The descriptor's refusals and names, and the streaming reference on cases
   whose answers are worked out by hand. Every comparison is on exact bits.
   Each known-answer case also runs through [Mutant], a plain re-statement of
   the recurrence with one defect switched on, which must turn that case red:
   a case a wrong evaluator passes would prove nothing. *)

module A = Attention_block
module S = A.Strides

let r32 = F32_math.round32
let canonical_nan = Int32.float_of_bits 0x7FC00000l
let bits x = Int32.bits_of_float x
let hex x = Printf.sprintf "%08lx" (bits x)
let strides batch head row col = { S.batch; head; row; col }

(* A dense single-head descriptor: Q [tq][d], K and V [tk][d], mask [tq][tk],
   O [tq][d]. *)
let dense ?(mask = false) ?scale ~tq ~tk ~d ~bk () =
  let tq = Int64.of_int tq and tk = Int64.of_int tk and d = Int64.of_int d in
  let scale = match scale with Some s -> s | None -> A.Scale.default ~d in
  Result.get_ok
    (Err.payload
       (A.make ~batch:1L ~heads:1L ~tq ~tk ~d ~q:(strides 0L 0L d 1L)
          ~k:(strides 0L 0L d 1L) ~v:(strides 0L 0L d 1L)
          ~o:(strides 0L 0L d 1L)
          ?mask:(if mask then Some (strides 0L 0L tk 1L) else None)
          ~scale ~block_keys:(Int64.of_int bk) ()))

let get a o = a.(Int64.to_int o)

let eval t ~q ~k ~v ~mask =
  let out = Array.make (Int64.to_int (Int64.mul t.A.tq t.A.d)) nan in
  A.eval t ~q:(get q) ~k:(get k) ~v:(get v) ~mask:(get mask) ~write:(fun o x ->
      out.(Int64.to_int o) <- x);
  out

(* ---- the recurrence with a defect switched on ---------------------------- *)

module Mutant = struct
  type t =
    | Add_after  (** each block's product from zero, then added to the seed *)
    | Column_select
        (** [p]'s empty-state select keyed on the column, [m' = -inf] *)
    | No_canonical_nan  (** a NaN keeps its payload through arithmetic *)
    | Normalized_rounding
        (** the row's normalized probability rounded to bfloat16 *)

  let nan_of ~canon x = if Float.is_nan x && canon then canonical_nan else x

  (* dense, one head; returns [tq * d] *)
  let eval ?mutant ~bk ~scale ~tq ~tk ~d ~q ~k ~v ?mask () =
    let canon = mutant <> Some No_canonical_nan in
    let rn x = if Float.is_nan x then nan_of ~canon x else r32 x in
    let fmax a b =
      if Float.is_nan a then if Float.is_nan b then canonical_nan else b
      else if Float.is_nan b then a
      else if a > b then a
      else if b > a then b
      else if Float.sign_bit a then b
      else a
    in
    let exp x = if Float.is_nan x then x else F32_math.Value.exp x in
    let pairs ~seed ~n xs ys =
      (* the engine's bfloat16 sum: two steps rounded to odd, added to nearest *)
      let acc = ref seed in
      let kk = ref 0 in
      while !kk < n do
        let x1, y1 =
          if !kk + 1 < n then (xs (!kk + 1), ys (!kk + 1)) else (0., 0.)
        in
        let sum =
          rn (!acc +. F32_math.add_odd32 (xs !kk *. ys !kk) (x1 *. y1))
        in
        acc := if sum = 0. then 0. else sum;
        kk := !kk + 2
      done;
      !acc
    in
    let out = Array.make (tq * d) 0. in
    for i = 0 to tq - 1 do
      let m = ref Float.neg_infinity and l = ref 0. in
      let acc = Array.make d 0. in
      let blocks = (tk + bk - 1) / bk in
      for j = 0 to blocks - 1 do
        let k0 = j * bk in
        let vk = min bk (tk - k0) in
        let z =
          Array.init vk (fun r ->
              let s =
                pairs ~seed:0.0 ~n:d
                  (fun f -> k.(((k0 + r) * d) + f))
                  (fun f -> q.((i * d) + f))
              in
              let sc = rn (s *. scale) in
              match mask with
              | None -> sc
              | Some mk -> rn (sc +. mk.((i * tk) + k0 + r)))
        in
        let m' = Array.fold_left fmax !m z in
        let a = if m' = Float.neg_infinity then 1. else exp (rn (!m -. m')) in
        let p =
          Array.map
            (fun zr ->
              let empty =
                if mutant = Some Column_select then m' = Float.neg_infinity
                else zr = Float.neg_infinity
              in
              if empty then 0. else exp (rn (zr -. m')))
            z
        in
        let s = Array.fold_left (fun s x -> rn (s +. x)) 0. p in
        let pb =
          Array.init bk (fun r ->
              if r < vk then Half.Bf16.to_float (Half.Bf16.of_float p.(r))
              else 0.)
        in
        let n = bk in
        for f = 0 to d - 1 do
          let seed = rn (a *. acc.(f)) in
          let xs kk = pb.(kk)
          and ys kk = if kk < vk then v.(((k0 + kk) * d) + f) else 0. in
          acc.(f) <-
            (if mutant = Some Add_after then rn (seed +. pairs ~seed:0. ~n xs ys)
             else pairs ~seed ~n xs ys)
        done;
        m := m';
        l := rn (rn (!l *. a) +. s)
      done;
      for f = 0 to d - 1 do
        out.((i * d) + f) <- (if !l = 0. then 0. else rn (acc.(f) /. !l))
      done
    done;
    (match mutant with
    | Some Normalized_rounding ->
        (* one block only: [p / l] rounded, then summed against V *)
        assert (tk <= bk);
        for i = 0 to tq - 1 do
          let z =
            Array.init tk (fun r ->
                let s =
                  pairs ~seed:0.0 ~n:d
                    (fun f -> k.((r * d) + f))
                    (fun f -> q.((i * d) + f))
                in
                let sc = r32 (s *. scale) in
                match mask with
                | None -> sc
                | Some mk -> r32 (sc +. mk.((i * tk) + r)))
          in
          let m = Array.fold_left fmax Float.neg_infinity z in
          let p = Array.map (fun zr -> exp (r32 (zr -. m))) z in
          let l = Array.fold_left (fun s x -> r32 (s +. x)) 0. p in
          let pb =
            Array.init bk (fun r ->
                if r < tk then
                  Half.Bf16.to_float (Half.Bf16.of_float (r32 (p.(r) /. l)))
                else 0.)
          in
          for f = 0 to d - 1 do
            out.((i * d) + f) <-
              pairs ~seed:0. ~n:bk
                (fun kk -> pb.(kk))
                (fun kk -> if kk < tk then v.((kk * d) + f) else 0.)
          done
        done
    | _ -> ());
    out
end

(* Bits as hex words; a run of one word, as a count. *)
let show a =
  if Array.length a > 4 && Array.for_all (fun x -> bits x = bits a.(0)) a then
    Printf.sprintf "%s x%d" (hex a.(0)) (Array.length a)
  else String.concat " " (Array.to_list (Array.map hex a))

(* One known-answer case: the library, the unmutated statement and each named
   mutant on the same inputs, all as bits. *)
let case ?(mutants = []) ?(unchanged = []) name ~bk ~scale ~tq ~tk ~d ~q ~k ~v
    ?mask ~expect () =
  let t = dense ~mask:(mask <> None) ~scale ~tq ~tk ~d ~bk () in
  let zero_mask = Array.make (tq * tk) 0. in
  let lib =
    eval t ~q ~k ~v ~mask:(match mask with Some m -> m | None -> zero_mask)
  in
  let stmt = Mutant.eval ~bk ~scale ~tq ~tk ~d ~q ~k ~v ?mask () in
  let ok a = show a = show (Array.of_list expect) in
  Fmt.pr "%s: %s%s@." name (show lib)
    (if ok lib && ok stmt then "" else "  <-- NOT the expected answer");
  List.iter
    (fun (mname, mutant) ->
      let m = Mutant.eval ~mutant ~bk ~scale ~tq ~tk ~d ~q ~k ~v ?mask () in
      Fmt.pr "  mutant %s: %s@." mname
        (match (ok m, List.mem mname unchanged) with
        | true, true -> "unchanged, as it must be"
        | true, false -> "STILL PASSES (vacuous)"
        | false, false -> show m
        | false, true -> "CHANGED (it should not)"))
    mutants

let zeros n = Array.make n 0.
let ones n = Array.make n 1.
let neg_inf = Float.neg_infinity

(* ---- the descriptor ------------------------------------------------------- *)

let%expect_test "refusals, names and symbols" =
  let s = strides 0L 0L 4L 1L in
  let mk ?(batch = 1L) ?(tk = 8L) ?(q = s) ?(scale = 0.5) ?(block_keys = 8L)
      ?mask () =
    A.make ~batch ~heads:1L ~tq:4L ~tk ~d:4L ~q ~k:s ~v:s ~o:s ?mask ~scale
      ~block_keys ()
  in
  let refuse r =
    match Err.payload r with
    | Ok t -> Fmt.pr "ok %s@." (A.name t)
    | Error e -> Fmt.pr "%a@." A.pp_error e
  in
  refuse (mk ~batch:0L ());
  refuse (mk ~tk:0L ());
  refuse (mk ~q:(strides 0L (-1L) 4L 1L) ());
  refuse (mk ~scale:0.1 ());
  refuse (mk ~scale:Float.nan ());
  refuse (mk ~scale:Float.infinity ());
  refuse (mk ~block_keys:0L ());
  refuse (mk ~block_keys:6L ());
  refuse (mk ~block_keys:(-4L) ());
  refuse (mk ~mask:(strides 0L 0L 0L (-2L)) ());
  refuse (mk ~mask:(strides 0L 0L 8L 1L) ());
  [%expect
    {|
    attention extent batch = 0
    attention extent tk = 0
    attention stride q.head = -1
    attention scale 0x1.999999999999ap-4, not a finite binary32
    attention scale nan, not a finite binary32
    attention scale infinity, not a finite binary32
    attention block of 0 keys, not a positive multiple of four
    attention block of 6 keys, not a positive multiple of four
    attention block of -4 keys, not a positive multiple of four
    attention stride mask.col = -2
    ok attend.b1.h1.q4.k8.d4.sq0_0_4_1.sk0_0_4_1.sv0_0_4_1.so0_0_4_1.m0_0_8_1.x3f000000.n8 |}]

let%expect_test "names round-trip" =
  let scales = [ 0.5; A.Scale.default ~d:64L; A.Scale.explicit 0.3; -2.0 ] in
  let n = ref 0 and bad = ref 0 in
  List.iter
    (fun scale ->
      List.iter
        (fun mask ->
          List.iter
            (fun (batch, heads, tq, tk, d, bk) ->
              let st a b c e = strides a b c e in
              match
                Err.payload
                  (A.make ~batch ~heads ~tq ~tk ~d ~q:(st 0L 3L d 1L)
                     ~k:(st (Int64.mul heads 7L) 0L d 1L)
                     ~v:(st 5L 0L 1L tk) ~o:(st 1L 2L d 1L)
                     ?mask:(if mask then Some (st 0L 0L 0L 1L) else None)
                     ~scale ~block_keys:bk ())
              with
              | Error _ -> incr bad
              | Ok t ->
                  incr n;
                  if A.of_name (A.name t) <> Some t then (
                    incr bad;
                    Fmt.pr "no round trip: %s@." (A.name t));
                  if
                    String.length (A.symbol t) <> 27 || A.symbol t <> A.symbol t
                  then incr bad)
            [
              (1L, 1L, 1L, 1L, 2L, 4L);
              (2L, 3L, 41L, 61L, 64L, 32L);
              (1L, 12L, 9L, 29L, 32L, 16L);
              (4L, 1L, 1024L, 1024L, 128L, 64L);
            ])
        [ false; true ])
    scales;
  Fmt.pr "%d descriptors, %d failures@." !n !bad;
  let a = dense ~tq:9 ~tk:29 ~d:64 ~bk:32 ()
  and b = dense ~tq:9 ~tk:29 ~d:64 ~bk:16 () in
  Fmt.pr "distinct symbols: %b; %s@." (A.symbol a <> A.symbol b) (A.symbol a);
  Fmt.pr "%s@." (A.name a);
  List.iter
    (fun s -> Fmt.pr "%S: %b@." s (A.of_name s = None))
    [ ""; "attend"; "attend.b1.h1"; "contract.f32.b1.m1.n1.k1" ];
  [%expect
    {|
    32 descriptors, 0 failures
    distinct symbols: true; mlt_attend_29f30d4daa709c9a
    attend.b1.h1.q9.k29.d64.sq0_0_64_1.sk0_0_64_1.sv0_0_64_1.so0_0_64_1.nomask.x3e000000.n32
    "": true
    "attend": true
    "attend.b1.h1": true
    "contract.f32.b1.m1.n1.k1": true |}]

(* ---- bfloat16 rounding ---------------------------------------------------- *)

let%expect_test "bf16_round is Half.Bf16.of_float on [0, 1] and NaN" =
  let los =
    [ 0; 1; 0x7FFE; 0x7FFF; 0x8000; 0x8001; 0xFFFE; 0xFFFF; 0x1234; 0xA5A5 ]
    @ List.init 54 (fun i -> i * 1217 land 0xFFFF)
  in
  let checked = ref 0 and bad = ref 0 in
  for hi = 0 to 0x3F80 do
    List.iter
      (fun lo ->
        if hi < 0x3F80 || lo = 0 then (
          let x =
            Int32.float_of_bits
              (Int32.logor
                 (Int32.shift_left (Int32.of_int hi) 16)
                 (Int32.of_int lo))
          in
          incr checked;
          if
            bits (A.bf16_round x)
            <> bits (Half.Bf16.to_float (Half.Bf16.of_float x))
          then incr bad))
      los
  done;
  List.iter
    (fun x ->
      incr checked;
      if
        bits (A.bf16_round x)
        <> bits (Half.Bf16.to_float (Half.Bf16.of_float x))
      then incr bad)
    [
      canonical_nan;
      Int32.float_of_bits 0x7FFFFFFFl;
      Int32.float_of_bits 0xFFC00000l;
      Float.infinity;
    ];
  Fmt.pr "%d patterns, %d differences@." !checked !bad;
  [%expect {| 1040390 patterns, 0 differences |}]

(* ---- known answers --------------------------------------------------------- *)

let%expect_test "seeded accumulation: add-after-the-block differs" =
  (* Zero scores, 64 keys in two blocks of 32, d = 2; every p and a is 1.
     Values 2^20 for the first 32 keys, 1 for the next four, 0 after. The
     first block leaves 2^25; seeded, each pair of the second adds 2, a tie that
     rounds back down at that magnitude (ulp 4), so the sum stays 2^25 and the
     output is 2^25 / 64 = 2^19. From zero then added, the block's four ones
     sum to 4 and the total is 2^25 + 4: 524288.0625. *)
  let tk = 64 and d = 2 in
  let v =
    Array.init (tk * d) (fun i ->
        let key = i / d in
        if key < 32 then ldexp 1. 20 else if key < 36 then 1. else 0.)
  in
  case "2^20 seeded" ~bk:32 ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
    ~k:(zeros (tk * d))
    ~v
    ~mutants:[ ("add_after", Mutant.Add_after) ]
    ~expect:[ ldexp 1. 19; ldexp 1. 19 ]
    ();
  [%expect
    {|
    2^20 seeded: 49000000 49000000
      mutant add_after: 49000001 49000001 |}]

let%expect_test "the rounded quantity is the unnormalized exponential" =
  (* Zero scores, three keys, unit values: p = 1 each, sum 3, 3 / 3 = 1. Rounding
     the normalized 1/3 to bfloat16 (0.333984375) three times gives
     1.001953125. *)
  let d = 2 and tk = 3 in
  case "tk 3 unit" ~bk:4 ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
    ~k:(zeros (tk * d))
    ~v:(ones (tk * d))
    ~mutants:[ ("normalized_rounding", Mutant.Normalized_rounding) ]
    ~expect:[ 1.; 1. ] ();
  [%expect
    {|
    tk 3 unit: 3f800000 3f800000
      mutant normalized_rounding: 3f804000 3f804000 |}]

let%expect_test "the key block is part of the grant" =
  (* Zero Q and K, 32 keys, d = 64; mask 0 on the first 16 keys and 1 on the
     last 16; V = 1 on the first 16 keys, 0 on the last 16. One block of 32
     rounds exp(-1) to bfloat16 (0.3671875) before the product: numerator
     16 * 0.3671875 = 5.875, output 0.26843557. Two blocks of 16 accumulate the
     unit probabilities first and rescale in binary32: numerator
     16 * exp(-1) = 5.88607120513916, output 0.26894143. *)
  let tk = 32 and d = 64 in
  let mask = Array.init tk (fun r -> if r < 16 then 0. else 1.) in
  let v = Array.init (tk * d) (fun i -> if i / d < 16 then 1. else 0.) in
  List.iter
    (fun (bk, answer) ->
      case (Fmt.str "bk %d" bk) ~bk ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
        ~k:(zeros (tk * d))
        ~v ~mask
        ~expect:(List.init d (fun _ -> answer))
        ())
    [ (32, r32 0.26843557); (16, r32 0.26894143) ];
  [%expect {|
    bk 32: 3e897063 x64
    bk 16: 3e89b2b1 x64 |}]

let%expect_test "an empty state: masked prefixes and columns" =
  (* Eight keys in two blocks of four, d = 2, zero scores, three queries.
     Query 0 sees every key, query 1 only the second block, query 2 none.
     V = 100 on the first block's keys, 1 on the second's:
     query 0 = (4 * 100 + 4 * 1) / 8 = 50.5, query 1 = 1, query 2 = +0. *)
  let tk = 8 and d = 2 and tq = 3 in
  let v = Array.init (tk * d) (fun i -> if i / d < 4 then 100. else 1.) in
  let mask =
    Array.init (tq * tk) (fun i ->
        let qi = i / tk and key = i mod tk in
        match qi with
        | 0 -> 0.
        | 1 -> if key < 4 then neg_inf else 0.
        | _ -> neg_inf)
  in
  case "masked prefixes" ~bk:4 ~scale:1. ~tq ~tk ~d
    ~q:(zeros (tq * d))
    ~k:(zeros (tk * d))
    ~v ~mask ~mutants:[]
    ~expect:[ 50.5; 50.5; 1.; 1.; 0.; 0. ]
    ();
  (* the only valid key of a column is the last of the last block *)
  let mask = Array.init tk (fun key -> if key = 7 then 0. else neg_inf) in
  let v = Array.init (tk * d) (fun i -> float_of_int ((i / d) + 1)) in
  case "only the last key" ~bk:4 ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
    ~k:(zeros (tk * d))
    ~v ~mask ~expect:[ 8.; 8. ] ();
  [%expect
    {|
    masked prefixes: 424a0000 424a0000 3f800000 3f800000 00000000 00000000
    only the last key: 41000000 41000000 |}]

let%expect_test "NaN scores make their column NaN" =
  let d = 2 and tk = 8 in
  let nan_ = canonical_nan in
  let v = ones (tk * d) in
  let column ?unchanged name mask =
    case ?unchanged name ~bk:4 ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
      ~k:(zeros (tk * d))
      ~v ~mask
      ~mutants:[ ("column_select", Mutant.Column_select) ]
      ~expect:[ nan_; nan_ ] ()
  in
  let all x = Array.make tk x in
  let with_at l x =
    let a = all x in
    List.iter (fun (i, y) -> a.(i) <- y) l;
    a
  in
  column "NaN with -inf only" (with_at [ (0, nan_) ] neg_inf);
  column "all NaN" (all nan_);
  column "NaN in a masked prefix, then a finite block"
    (Array.init tk (fun i ->
         if i = 0 then nan_ else if i < 4 then neg_inf else 0.));
  column ~unchanged:[ "column_select" ] "NaN mixed with finite"
    (with_at [ (5, nan_) ] 0.);
  [%expect
    {|
    NaN with -inf only: 7fc00000 7fc00000
      mutant column_select: 00000000 00000000
    all NaN: 7fc00000 7fc00000
      mutant column_select: 00000000 00000000
    NaN in a masked prefix, then a finite block: 7fc00000 7fc00000
      mutant column_select: 3f800000 3f800000
    NaN mixed with finite: 7fc00000 7fc00000
      mutant column_select: unchanged, as it must be |}]

let%expect_test "a payload NaN mask element gives the canonical NaN" =
  let d = 2 and tk = 4 in
  let payload = Int32.float_of_bits 0x7FFFFFFFl in
  case "payload NaN" ~bk:4 ~scale:1. ~tq:1 ~tk ~d ~q:(zeros d)
    ~k:(zeros (tk * d))
    ~v:(ones (tk * d))
    ~mask:[| 0.; payload; 0.; 0. |]
    ~mutants:[ ("no_canonical_nan", Mutant.No_canonical_nan) ]
    ~expect:[ canonical_nan; canonical_nan ]
    ();
  [%expect
    {|
    payload NaN: 7fc00000 7fc00000
      mutant no_canonical_nan: 7fffffff 7fffffff |}]

let%expect_test "a negative-zero seed meets a short last block" =
  (* Zero Q and K, 34 keys, blocks of 32, d = 64; mask 0 on the first 32 keys
     and 200 on the last two. V is -1 on the first 32 keys and -0 on the last
     two in feature 0. Block 0 leaves -32; block 1 has m = 200, a = exp(-200) =
     +0, seed RN(+0 * -32) = -0; the two valid keys add -0 and -0 and the 30
     padded keys add +0 pairs. The engine's adder gives +0 for a zero sum
     (measured), so the output is +0 whether the padded pairs run or not. *)
  let tk = 34 and d = 64 in
  let mask = Array.init tk (fun r -> if r < 32 then 0. else 200.) in
  let v =
    Array.init (tk * d) (fun i ->
        if i mod d = 0 then if i / d < 32 then -1. else -0. else 1.)
  in
  let t = dense ~mask:true ~tq:1 ~tk ~d ~bk:32 () in
  let out = eval t ~q:(zeros d) ~k:(zeros (tk * d)) ~v ~mask in
  let stmt =
    Mutant.eval ~bk:32 ~scale:(A.Scale.default ~d:64L) ~tq:1 ~tk ~d ~q:(zeros d)
      ~k:(zeros (tk * d))
      ~v ~mask ()
  in
  Fmt.pr "feature 0: library %s, statement %s@." (hex out.(0)) (hex stmt.(0));
  [%expect {| feature 0: library 00000000, statement 00000000 |}]

let%expect_test "a subnormal probability" =
  (* exp(-90) is subnormal in binary32; its low half passes 0x8000, so the
     bfloat16 rounding carries up. The library and the statement (which rounds
     with Half) agree on every bit. *)
  let d = 2 and tk = 4 in
  let mask = [| 0.; -90.; -90.; -90. |] in
  let v = Array.init (tk * d) (fun i -> float_of_int ((i / d) + 1)) in
  let p = F32_math.Value.exp (r32 (-90.)) in
  Fmt.pr "p = %h subnormal: %b@." p (p < 0x1p-126 && p > 0.);
  let t = dense ~mask:true ~tq:1 ~tk ~d ~bk:4 () in
  let lib = eval t ~q:(zeros d) ~k:(zeros (tk * d)) ~v ~mask in
  let stmt =
    Mutant.eval ~bk:4 ~scale:(A.Scale.default ~d:2L) ~tq:1 ~tk ~d ~q:(zeros d)
      ~k:(zeros (tk * d))
      ~v ~mask ()
  in
  Fmt.pr "%s | %s@." (show lib) (show stmt);
  [%expect
    {|
    p = 0x1.1d85p-130 subnormal: true
    3f800001 3f800001 | 3f800001 3f800001 |}]

(* ---- strides and random data ----------------------------------------------- *)

let lcg = ref 12345

let next () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  !lcg

(* bfloat16 values with a spread: small integers over four, a few large *)
let bf16_value () =
  let n = next () in
  Half.Bf16.to_float
    (Half.Bf16.of_float
       (if n mod 29 = 0 then float_of_int ((n / 29 mod 61) - 30)
        else float_of_int ((n / 7 mod 33) - 16) /. 8.))

let mask_value () =
  match next () mod 11 with
  | 0 | 1 -> neg_inf
  | 2 -> 3.5
  | n -> float_of_int (n - 6) /. 4.

let%expect_test "the reference is the statement of it, on ragged shapes" =
  List.iter
    (fun (tq, tk, d, bk, masked) ->
      let q = Array.init (tq * d) (fun _ -> bf16_value ())
      and k = Array.init (tk * d) (fun _ -> bf16_value ())
      and v = Array.init (tk * d) (fun _ -> bf16_value ())
      and mask = Array.init (tq * tk) (fun _ -> mask_value ()) in
      let scale = A.Scale.default ~d:(Int64.of_int d) in
      let t = dense ~mask:masked ~tq ~tk ~d ~bk () in
      let lib = eval t ~q ~k ~v ~mask in
      let stmt =
        Mutant.eval ~bk ~scale ~tq ~tk ~d ~q ~k ~v
          ?mask:(if masked then Some mask else None)
          ()
      in
      Fmt.pr "%dx%d d%d bk%d%s: %s@." tq tk d bk
        (if masked then " masked" else "")
        (if Array.for_all2 (fun a b -> bits a = bits b) lib stmt then "equal"
         else "DIFFERENT"))
    [
      (9, 29, 64, 32, true);
      (41, 61, 64, 32, true);
      (16, 36, 32, 32, false);
      (32, 32, 32, 16, true);
      (5, 70, 8, 8, true);
      (3, 1, 2, 4, false);
    ];
  [%expect
    {|
    9x29 d64 bk32 masked: equal
    41x61 d64 bk32 masked: equal
    16x36 d32 bk32: equal
    32x32 d32 bk16 masked: equal
    5x70 d8 bk8 masked: equal
    3x1 d2 bk4: equal |}]

let%expect_test "strides: batches, heads shared by K and V, a transposed K" =
  let batch = 2 and heads = 3 and tq = 9 and tk = 29 and d = 8 and bk = 16 in
  let q = Array.init (batch * heads * tq * d) (fun _ -> bf16_value ())
  and k = Array.init (batch * tk * d) (fun _ -> bf16_value ())
  and v = Array.init (batch * tk * d) (fun _ -> bf16_value ()) in
  (* K stored feature-leading per batch: element (key, f) at f * tk + key *)
  let k_t =
    Array.init
      (batch * tk * d)
      (fun i ->
        let b = i / (tk * d) and r = i mod (tk * d) in
        let f = r / tk and key = r mod tk in
        k.((b * tk * d) + (key * d) + f))
  in
  let scale = A.Scale.default ~d:(Int64.of_int d) in
  let i64 = Int64.of_int in
  List.iter
    (fun (name, mrows, mcols) ->
      let mask = Array.init (batch * mrows * mcols) (fun _ -> mask_value ()) in
      let desc =
        Result.get_ok
          (Err.payload
             (A.make ~batch:(i64 batch) ~heads:(i64 heads) ~tq:(i64 tq)
                ~tk:(i64 tk) ~d:(i64 d)
                ~q:(strides (i64 (heads * tq * d)) (i64 (tq * d)) (i64 d) 1L)
                ~k:(strides (i64 (tk * d)) 0L 1L (i64 tk))
                ~v:(strides (i64 (tk * d)) 0L (i64 d) 1L)
                ~o:(strides (i64 (heads * tq * d)) (i64 (tq * d)) (i64 d) 1L)
                ~mask:
                  (strides
                     (i64 (mrows * mcols))
                     0L
                     (if mrows = 1 then 0L else i64 mcols)
                     (if mcols = 1 then 0L else 1L))
                ~scale ~block_keys:(i64 bk) ()))
      in
      let out = Array.make (batch * heads * tq * d) nan in
      A.eval desc ~q:(get q) ~k:(get k_t) ~v:(get v) ~mask:(get mask)
        ~write:(fun o x -> out.(Int64.to_int o) <- x);
      let same = ref true in
      for b = 0 to batch - 1 do
        for h = 0 to heads - 1 do
          let qh = Array.sub q (((b * heads) + h) * tq * d) (tq * d)
          and kh = Array.sub k (b * tk * d) (tk * d)
          and vh = Array.sub v (b * tk * d) (tk * d)
          and mh =
            Array.init (tq * tk) (fun i ->
                let r = if mrows = 1 then 0 else i / tk
                and c = if mcols = 1 then 0 else i mod tk in
                mask.((b * mrows * mcols) + (r * mcols) + c))
          in
          let want =
            eval (dense ~mask:true ~tq ~tk ~d ~bk ()) ~q:qh ~k:kh ~v:vh ~mask:mh
          in
          let got = Array.sub out (((b * heads) + h) * tq * d) (tq * d) in
          if not (Array.for_all2 (fun a b -> bits a = bits b) want got) then
            same := false
        done
      done;
      Fmt.pr "%s: %s@." name
        (if !same then "equal to the dense runs" else "DIFFERENT"))
    [
      ("mask [tq, tk]", tq, tk); ("mask [1, tk]", 1, tk); ("mask [tq, 1]", tq, 1);
    ];
  [%expect
    {|
    mask [tq, tk]: equal to the dense runs
    mask [1, tk]: equal to the dense runs
    mask [tq, 1]: equal to the dense runs |}]
