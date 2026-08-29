let%expect_test "vec6: offset, numel, pp" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:2 in
  Format.printf "%a @@ %a -> off %a, numel %a@." Vec6.pp_shape s Vec6.pp_coord c
    Dim.pp (Vec6.offset s c) Dim.pp (Vec6.numel s);
  [%expect {| [H=2 W=2 C=3] @ (1,0,2) -> off 8, numel 12 |}]

let%expect_test "vec6: in_bounds checks each axis upper bound" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let probe ~h ~w ~c = Vec6.in_bounds s (Vec6.coord ~n:0 ~t:0 ~d:0 ~h ~w ~c) in
  Format.printf "%b %b %b@." (probe ~h:0 ~w:0 ~c:2) (probe ~h:0 ~w:0 ~c:3)
    (probe ~h:5 ~w:9 ~c:2);
  [%expect {| true false false |}]

let%expect_test "vec6: iter visits dense offsets 0..numel-1 (C innermost)" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  Vec6.iter s (fun c -> Format.printf "%d " (Vec6.offset s c :> int));
  Format.printf "@.";
  [%expect {| 0 1 2 3 4 5 6 7 |}]

(* [numel_bounded]'s divide-before-multiply fold, checked against
   [Kernel.Limits.Hard.numel] (2^31, exclusive) directly rather than through
   any importer -- this is the primitive op3-impl.md's F1 lifts everything
   else from, and running under both [best] and [js] ([test/native/dune]'s
   [(modes best js)]) is the whole point: an [int] equality between two
   unchecked [Vec6.numel] results is exactly what wraps two DIFFERENT
   products to the same 32-bit value under js_of_ocaml. *)
let pp_bounded ppf = function
  | Error e -> (
      match Err.Error.kind e with
      | `Numel_over_limit b -> Fmt.pf ppf "Error %a" Vec6.Numel_bound.pp b)
  | Ok n -> Fmt.pf ppf "Ok %Ld" n

let numel_bounded s = Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel s

let%expect_test "vec6: numel_bounded accepts an ordinary shape" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  Format.printf "%a@." pp_bounded (numel_bounded s);
  [%expect {| Ok 24 |}]

(* The fencepost: [h*w = 65536*32768 = 2^31] EXACTLY, which the hard contract
   excludes ("stays BELOW 2^31") and which [Vec6.numel] cannot represent as a
   js_of_ocaml [int] at all. An earlier revision of this primitive's bound was
   inclusive and accepted exactly this count -- nothing else in the suite
   would have caught that. *)
let%expect_test "vec6: numel_bounded rejects exactly 2^31 (the fencepost)" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:65536 ~w:32768 ~c:1 in
  Format.printf "%a@." pp_bounded (numel_bounded s);
  [%expect
    {|
    Error axis W: 65536 elements so far times extent 32768 reaches the maximum of 2147483648 |}]

(* [65536,65536] = 2^32 and [65536,131072] = 2^33 are DIFFERENT products that
   both wrap to 0 as a 32-bit [int] -- the case a plain [int] equality between
   two unchecked [Vec6.numel] results accepts under jsoo. [numel_bounded]
   rejects both, and the witnesses differ (extent 65536 vs 131072): it never
   folds either product into a single number to compare. *)
let%expect_test "vec6: numel_bounded rejects 2^32, distinctly from 2^33" =
  let s32 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:65536 ~w:65536 ~c:1 in
  let s33 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:65536 ~w:131072 ~c:1 in
  Format.printf "%a@." pp_bounded (numel_bounded s32);
  Format.printf "%a@." pp_bounded (numel_bounded s33);
  [%expect
    {|
    Error axis W: 65536 elements so far times extent 65536 reaches the maximum of 2147483648
    Error axis W: 65536 elements so far times extent 131072 reaches the maximum of 2147483648 |}]
