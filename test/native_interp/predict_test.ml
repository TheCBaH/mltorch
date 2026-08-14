(* [Native_predict.top_predictions] — the contract checks and the numerics.

   These are the cases the .pt2 fixtures cannot reach. mobilenet_v3_small always
   hands over exactly one well-shaped [1; 1000] tensor of finite, moderate
   logits, so every rejection arm and every floating-point edge below is
   unreachable from `make jsoo.pt2.run`. Hand-built tensors are the only way to
   see them at all. Runs under node too — see the [js] mode in dune. *)

(* A [classes]-long logits tensor at the shape a classifier tail really
   produces: PT2's [1; classes] right-aligns into Vec6 as [~w:1 ~c:classes]. *)
let logits values =
  let classes = Array.length values in
  Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:classes)
    (fun coord -> values.(Dim.to_int (Vec6.get coord Axis.C)))

(* [Core.Pretty.err_result] renders the error KIND, not [Err.Error.pp]'s full
   wrapper: the detection backtrace is exactly the fragile output ppx_expect
   warns about, and it differs between the native and node runs of this suite. *)
let pp_failure ppf e = Fmt.pf ppf "@[<h>error: %a@]" Native_predict.pp_error e

let pp_top ppf top =
  Fmt.pf ppf "@[<h>%a@]"
    (Fmt.list ~sep:(Fmt.any "  ") (fun ppf (i, p) -> Fmt.pf ppf "%d:%.6f" i p))
    top

let show outputs k =
  Fmt.pr "%a@."
    (Core.Pretty.err_result ~ok:pp_top ~error:pp_failure)
    (Native_predict.top_predictions outputs k)

(* [k] is validated before the classes count, so a negative [k] cannot pass the
   [classes >= k] test by being small. *)
let%expect_test "k must be positive" =
  List.iter (fun k -> show [ logits [| 1.; 2.; 3. |] ] k) [ 0; -1; 1 ];
  [%expect
    {|
    error: top-k needs k >= 1, got 0
    error: top-k needs k >= 1, got -1
    2:0.665241 |}]

(* The graph must have reduced to exactly one tensor. Zero and two both fail,
   and the payload reports which. *)
let%expect_test "exactly one output tensor" =
  let one = logits [| 1.; 2. |] in
  List.iter (fun outputs -> show outputs 1) [ []; [ one; one ] ];
  [%expect
    {|
    error: expected exactly one output tensor, got 0
    error: expected exactly one output tensor, got 2 |}]

(* A rank-two [2; classes] PT2 result is [~w:2 ~c:classes] — the batch lands on
   W, NOT on N, because sizes right-align into Vec6. The [~n:2] case is the same
   rejection reached through a different axis, and is here so the check is not
   accidentally specialised to whichever axis the real models happen to use. *)
let%expect_test "one batch of class logits only" =
  let at shape = show [ Tensor.materialize shape (fun _ -> 1.) ] 1 in
  at (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1000);
  at (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1000);
  at (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:1 ~c:4);
  [%expect
    {|
    error: not one batch of class logits: [W=2 C=1000]
    error: not one batch of class logits: [N=2 T=1 D=1 H=1 W=1 C=1000]
    error: not one batch of class logits: [H=3 W=1 C=4] |}]

(* Rejected rather than clamped, so the two runners agree: ATen's [topk] raises
   on k > n instead of returning a short list. *)
let%expect_test "fewer classes than requested" =
  show [ logits [| 1.; 2.; 3. |] ] 5;
  [%expect {| error: top-5 requested, only 3 classes |}]

(* Without this arm a +infinity maximum makes [l -. max] NaN for the maximum
   itself, and the whole softmax comes back NaN while still reporting Ok. *)
let%expect_test "non-finite logits are refused, first offender named" =
  show [ logits [| 1.; Float.nan; 3. |] ] 2;
  show [ logits [| 1.; 2.; Float.infinity |] ] 2;
  show [ logits [| Float.neg_infinity; 2.; Float.nan |] ] 2;
  [%expect
    {|
    error: non-finite logit at class 1: nan
    error: non-finite logit at class 2: infinity
    error: non-finite logit at class 0: -infinity |}]

(* Logits far from zero. A naive [exp l /. sum (exp l)] overflows to inf/inf =
   NaN here; subtracting the maximum first keeps every term in range. *)
let%expect_test "large logits stay finite and normalized" =
  let values = [| 1000.; 1001.; 999. |] in
  show [ logits values ] 3;
  Fmt.pr "%a@."
    (Core.Pretty.err_result
       ~ok:(fun ppf top ->
         Fmt.pf ppf "sum %.6f" (List.fold_left (fun a (_, p) -> a +. p) 0. top))
       ~error:pp_failure)
    (Native_predict.top_predictions [ logits values ] 3);
  [%expect {|
    1:0.665241  0:0.244728  2:0.090031
    sum 1.000000 |}]

(* The denominator spans every class, not the selected k: four equal logits give
   0.25 each, and the top two sum to 0.5 rather than to 1. Normalizing over the
   selection would print 0.500000 twice. The cases above all use k = classes and
   cannot tell the two apart. *)
let%expect_test "softmax normalizes over every class, not the selection" =
  show [ logits [| 1.; 1.; 1.; 1. |] ] 2;
  show [ logits [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |] ] 3;
  [%expect
    {|
    0:0.250000  1:0.250000
    0:0.125000  1:0.125000  2:0.125000 |}]

(* The case that pins ranking-by-logit. Both exp(-1001) and exp(-1000) underflow
   to exactly 0., so ranking the probabilities sees a tie and falls back to
   ascending index, returning class 1 ahead of class 2 — the wrong order. The
   logits themselves are ordered, so ranking them gets it right. *)
let%expect_test "underflowed probabilities do not reorder classes" =
  show [ logits [| 0.; -1001.; -1000. |] ] 3;
  [%expect {| 0:1.000000  2:0.000000  1:0.000000 |}]

(* Genuinely equal logits DO tie, and break by ascending class index so both
   backends agree on the listing. *)
let%expect_test "ties break by ascending class index" =
  show [ logits [| 2.; 5.; 5.; 2.; 5. |] ] 5;
  [%expect {|
    1:0.322625  2:0.322625  4:0.322625  0:0.016063  3:0.016063 |}]

(* A hand-checkable vector: e^0, e^1, e^2 over the same denominator. *)
let%expect_test "descending order, hand-checked probabilities" =
  show [ logits [| 0.; 2.; 1. |] ] 3;
  [%expect {| 1:0.665241  2:0.244728  0:0.090031 |}]
