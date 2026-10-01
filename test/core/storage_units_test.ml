(* [Core.Storage_units]: every constructor refuses what its domain excludes,
   and every operation refuses a result it cannot represent, near the [int64]
   edges as well as at zero. *)

open Core.Storage_units

let show pp r = Fmt.pr "%a@." (Core.Pretty.err_result ~ok:pp ~error:pp_error) r
let ok r = Err.or_raise ~pp_error r
let max = Int64.max_int
let size v = ok (Byte_size.of_int64 v)
let offset v = ok (Byte_offset.of_int64 v)
let alignment v = ok (Byte_alignment.of_int64 v)
let width v = ok (Element_bytes.of_int64 v)
let count v = ok (Element_count.of_int64 v)
let eoffset v = ok (Element_offset.of_int64 v)

let%expect_test "constructors: zero where legal, never negative" =
  show Byte_size.pp (Byte_size.of_int64 0L);
  show Byte_size.pp (Byte_size.of_int64 (-1L));
  show Byte_offset.pp (Byte_offset.of_int64 0L);
  show Byte_offset.pp (Byte_offset.of_int64 Int64.min_int);
  show Element_count.pp (Element_count.of_int64 0L);
  show Element_offset.pp (Element_offset.of_int64 (-7L));
  show Element_bytes.pp (Element_bytes.of_int64 0L);
  show Element_bytes.pp (Element_bytes.of_int64 3L);
  [%expect
    {|
    0
    invalid byte size -1: negative
    0
    invalid byte offset -9223372036854775808: negative
    0
    invalid element offset -7: negative
    invalid element width 0: zero
    3
    |}]

let%expect_test "alignment: a positive power of two" =
  List.iter
    (fun v -> show Byte_alignment.pp (Byte_alignment.of_int64 v))
    [ 1L; 64L; 4096L; 0x4000_0000_0000_0000L; 0L; -64L; 48L; max ];
  show Byte_alignment.pp (Byte_alignment.of_element_bytes (width 4L));
  show Byte_alignment.pp (Byte_alignment.of_element_bytes (width 3L));
  [%expect
    {|
    1
    64
    4096
    4611686018427387904
    invalid byte alignment 0: zero
    invalid byte alignment -64: negative
    invalid byte alignment 48: not a power of two
    invalid byte alignment 9223372036854775807: not a power of two
    4
    invalid element width 3: not a power of two
    |}]

let%expect_test "nonzero refinements" =
  show
    (Fmt.using Byte_size.Nonzero.to_size Byte_size.pp)
    (Byte_size.Nonzero.of_size Byte_size.zero);
  show
    (Fmt.using Byte_size.Nonzero.to_size Byte_size.pp)
    (Byte_size.Nonzero.of_size (size 1L));
  show
    (Fmt.using Element_count.Nonzero.to_count Element_count.pp)
    (Element_count.Nonzero.of_count Element_count.zero);
  show
    (Fmt.using Element_count.Nonzero.to_count Element_count.pp)
    (Element_count.Nonzero.of_count (count 5L));
  [%expect
    {|
    invalid byte size 0: zero
    1
    invalid element count 0: zero
    5
    |}]

let%expect_test "byte size arithmetic" =
  show Byte_size.pp (Byte_size.add (size 3L) (size 4L));
  show Byte_size.pp (Byte_size.add (size max) Byte_size.zero);
  show Byte_size.pp (Byte_size.add (size max) (size 1L));
  show Byte_size.pp (Byte_size.sub (size 4L) (size 4L));
  show Byte_size.pp (Byte_size.sub (size 3L) (size 4L));
  [%expect
    {|
    7
    9223372036854775807
    overflow: add 9223372036854775807, 1
    0
    underflow: subtract 3, 4
    |}]

let%expect_test "byte offsets: advance, align, distance" =
  show Byte_offset.pp (Byte_offset.advance (offset 10L) (size 6L));
  show Byte_offset.pp (Byte_offset.advance (offset max) (size 1L));
  show Byte_offset.pp (Byte_offset.align_up (offset 0L) (alignment 4096L));
  show Byte_offset.pp (Byte_offset.align_up (offset 1L) (alignment 64L));
  show Byte_offset.pp (Byte_offset.align_up (offset 64L) (alignment 64L));
  show Byte_offset.pp (Byte_offset.align_up (offset 65L) (alignment 64L));
  (* The naive [t + alignment - 1] overflows on the aligned value just below
     the ceiling; the remainder form does not. *)
  let top = Int64.sub max 63L in
  show Byte_offset.pp (Byte_offset.align_up (offset top) (alignment 64L));
  show Byte_offset.pp
    (Byte_offset.align_up (offset (Int64.succ top)) (alignment 64L));
  Fmt.pr "%b %b@."
    (Byte_offset.is_aligned (offset 128L) (alignment 64L))
    (Byte_offset.is_aligned (offset 96L) (alignment 64L));
  show Byte_size.pp (Byte_offset.distance ~from:(offset 4L) (offset 10L));
  show Byte_size.pp (Byte_offset.distance ~from:(offset 10L) (offset 4L));
  [%expect
    {|
    16
    overflow: advance 9223372036854775807, 1
    0
    64
    64
    128
    9223372036854775744
    overflow: align up 9223372036854775745, 64
    true false
    6
    underflow: distance 4, 10
    |}]

let%expect_test "padding a size to an alignment" =
  show Byte_size.pp (Byte_alignment.pad (size 0L) (alignment 64L));
  show Byte_size.pp (Byte_alignment.pad (size 16L) (alignment 64L));
  show Byte_size.pp (Byte_alignment.pad (size 4096L) (alignment 4096L));
  show Byte_size.pp (Byte_alignment.pad (size 4097L) (alignment 4096L));
  show Byte_size.pp
    (Byte_alignment.pad (size (Int64.sub max 62L)) (alignment 64L));
  [%expect
    {|
    0
    64
    4096
    8192
    overflow: align up 9223372036854775745, 64 |}]

let%expect_test "element conversions" =
  show Byte_size.pp (Element_count.to_bytes (count 100L) (width 4L));
  show Byte_size.pp
    (Element_count.to_bytes (count (Int64.div max 8L)) (width 8L));
  show Byte_size.pp
    (Element_count.to_bytes (count (Int64.succ (Int64.div max 8L))) (width 8L));
  show Element_count.pp (Element_count.of_bytes (size 400L) (width 4L));
  show Element_count.pp (Element_count.of_bytes (size 402L) (width 4L));
  show Byte_offset.pp (Element_offset.to_bytes (eoffset 16L) (width 2L));
  show Byte_offset.pp (Element_offset.to_bytes (eoffset max) (width 2L));
  show Element_offset.pp (Element_offset.of_bytes (offset 64L) (width 8L));
  show Element_offset.pp (Element_offset.of_bytes (offset 65L) (width 8L));
  [%expect
    {|
    400
    9223372036854775800
    overflow: scale 1152921504606846976, 8
    100
    not a whole number of elements: convert to elements 402, 4
    32
    overflow: scale 9223372036854775807, 2
    8
    not a whole number of elements: convert to elements 65, 8
    |}]

let%expect_test "element offsets: advance, distance" =
  show Element_offset.pp (Element_offset.advance (eoffset 3L) (count 4L));
  show Element_offset.pp (Element_offset.advance (eoffset max) (count 1L));
  show Element_count.pp
    (Element_offset.distance ~from:(eoffset 3L) (eoffset 7L));
  show Element_count.pp
    (Element_offset.distance ~from:(eoffset 7L) (eoffset 3L));
  [%expect
    {|
    7
    overflow: advance 9223372036854775807, 1
    4
    underflow: distance 3, 7
    |}]

let%expect_test "sizes: succ, pred, midpoint; offsets from the pool start" =
  show Byte_size.pp (Byte_size.succ (size 4L));
  show Byte_size.pp (Byte_size.succ (size max));
  show Byte_size.pp (Byte_size.pred (size 4L));
  show Byte_size.pp (Byte_size.pred Byte_size.zero);
  Fmt.pr "%a %a %a@." Byte_size.pp
    (Byte_size.midpoint (size 10L) (size 19L))
    Byte_size.pp
    (Byte_size.midpoint (size 19L) (size 10L))
    Byte_size.pp
    (Byte_size.midpoint (size (Int64.pred max)) (size max));
  Fmt.pr "%a %a@." Byte_offset.pp
    (Byte_offset.of_size (size 12L))
    Byte_size.pp
    (Byte_offset.to_size (offset 12L));
  [%expect
    {|
    5
    overflow: add 9223372036854775807, 1
    3
    underflow: subtract 0, 1
    14 14 9223372036854775806
    12 12
    |}]
