(* [Alignment_policy.default]: a cache line up to and including a page, a page
   above it, never below the payload's own alignment. *)

open Core.Storage_units

let units r = Err.or_raise ~pp_error r
let size v = units (Byte_size.of_int64 v)
let align v = units (Byte_alignment.of_int64 v)
let page = Byte_size.to_int64 Alignment_policy.page_size

let%expect_test "the size thresholds" =
  List.iter
    (fun v ->
      Fmt.pr "%Ld -> %a@." v Byte_alignment.pp
        (Alignment_policy.default (size v) ~payload_min:(align 1L)))
    [ 0L; 1L; Int64.pred page; page; Int64.succ page ];
  [%expect
    {|
    0 -> 64
    1 -> 64
    4095 -> 64
    4096 -> 64
    4097 -> 4096 |}]

let%expect_test "the payload's alignment is a floor" =
  List.iter
    (fun (v, m) ->
      Fmt.pr "%Ld, payload %Ld -> %a@." v m Byte_alignment.pp
        (Alignment_policy.default (size v) ~payload_min:(align m)))
    [ (16L, 8L); (16L, 128L); (Int64.succ page, 8192L) ];
  [%expect
    {|
    16, payload 8 -> 64
    16, payload 128 -> 128
    4097, payload 8192 -> 8192 |}]

(* A host request is a third floor: it strengthens the default and never
   weakens it. *)
let%expect_test "a host request can only strengthen" =
  List.iter
    (fun (v, host) ->
      let policy = Alignment_policy.with_host (align host) in
      Fmt.pr "%Ld, %a -> %a@." v Alignment_policy.pp policy Byte_alignment.pp
        (Alignment_policy.alignment policy (size v) ~payload_min:(align 4L)))
    [
      (16L, 8L); (16L, 256L); (Int64.succ page, 64L); (Int64.succ page, 65536L);
    ];
  Fmt.pr "standard: %a@." Byte_alignment.pp
    (Alignment_policy.alignment Alignment_policy.standard (size 16L)
       ~payload_min:(align 4L));
  [%expect
    {|
    16, standard, host 8 -> 64
    16, standard, host 256 -> 256
    4097, standard, host 64 -> 4096
    4097, standard, host 65536 -> 65536
    standard: 64 |}]
