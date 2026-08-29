(* scaled_dot_product_attention.default, hand-computed against ATen's
   `math` backend structure. Split from compute_test.ml. *)

open Compute_fixtures

(* ---- sdpa ------------------------------------------------------------------

   op8-impl.md commit 1 step 9: since [Direct] vs [Symbolic] cannot see a wrong
   formula (both run the same [Compute] functor), THIS is where the row is
   actually proven -- every case below is hand-computed against ATen's `math`
   backend structure (F3: split-sqrt scale, unconditional `_safe_softmax`),
   never against ATen itself.

   Mutation proof (CLAUDE.md: "prove the check can fail"), each applied to
   [Attention.Sdpa.Compute] in [lib/native/ops/attention.ml], run against this
   suite, OBSERVED changing at least one golden below, then reverted -- none
   of these mutations is left in the tree:
     - Q@K instead of Q@K^T (swapped which axis of [key] is the contraction
       index and which is the enumeration index);
     - scale applied after the softmax rather than split across the operands
       before the dot;
     - mask omitted;
     - mask sign reversed (subtracted instead of added);
     - the row max reduced over the wrong extent (E instead of Wk);
     - the denominator reduced over the wrong extent (E instead of Wk);
     - max-subtraction omitted (observed: [nan]/[inf] on the large-logits
       case, not merely a smaller numeric drift);
     - probabilities paired with the wrong value index (read [value] at the
       output's own row instead of the reduction variable);
     - D and H swapped;
     - the `_safe_softmax` guard removed (observed: [nan] on the all-`-inf`
       case, per F3 -- exactly the defect the guard exists to prevent).
   "scale applied once rather than split" is NOT a [%g]-rounded-printing
   mutation: [sqrt(scale)^2 = scale] in exact real arithmetic, so it agrees
   with the split form to 6 significant figures on ordinary inputs. It is
   proven separately below, bitwise, without touching [attention.ml] at all. *)

let run_sdpa params ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask
    =
  let module S = Attention.Sdpa.Compute (Direct) in
  eval_tensor
    (Attention.Sdpa.output_shape ~query_shape ~key_shape ~value_shape:key_shape
       ~mask_shape:(Some mask_shape))
    (S.pixel params ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask)

let no_mask = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let zero_mask shape = Tensor.materialize shape (fun _ -> 0.)

let%expect_test "Direct: sdpa — one query, one key (trivial attention = value)"
    =
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun c -> [| 1.; 2. |].(chan c)) in
  let key = Tensor.materialize key_shape (fun c -> [| 3.; 4. |].(chan c)) in
  let value = Tensor.materialize key_shape (fun c -> [| 7.; 8. |].(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=2] {7, 8} |}]

let%expect_test "Direct: sdpa — two keys, unequal scores" =
  (* q=[1,0]; k0=[1,0] (dot=1), k1=[0,1] (dot=0); scale=1 so score = dot.
     m=1, z=1+exp(-1); p0=1/z, p1=exp(-1)/z. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let query = Tensor.materialize query_shape (fun c -> [| 1.; 0. |].(chan c)) in
  let key =
    Tensor.materialize key_shape (fun c ->
        [| [| 1.; 0. |]; [| 0.; 1. |] |].(col c).(chan c))
  in
  let value =
    Tensor.materialize key_shape (fun c ->
        [| [| 10.; 20. |]; [| 30.; 40. |] |].(col c).(chan c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=2] {15.3788, 25.3788} |}]

let%expect_test "Direct: sdpa — default scale is 1/sqrt(head_dim)" =
  (* E=4: default scale = 0.5. q=[1,1,1,1], k0=q (dot=4 -> score=2), k1=0
     (dot=0 -> score=0). v0/v1 are constant vectors, so every output feature
     is the same softmax-weighted blend. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:4 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 0.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=4] {1.1192, 1.1192, 1.1192, 1.1192} |}]

let%expect_test "Direct: sdpa — explicit scale" =
  (* q=3, k0=1, k1=2, scale=2 (explicit) -> score = dot*scale = 6, 12. v0=5,
     v1=7. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 3.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 5. else 7.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 2.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=1] {6.99505} |}]

(* Mutation proof: "scale applied once rather than split" (F3) is NOT a
   mutation [%g]-rounded printing can catch -- [sqrt(scale)^2 = scale] in
   EXACT real arithmetic, so the two forms agree to 6 significant figures for
   ordinary inputs. They are still genuinely different floats: rounding at the
   [sqrt] and at each of the two multiplies means [dot(q*sf, k*sf)] and
   [dot(q,k)*scale] are almost never bit-identical (observed here: a ULP-scale
   divergence, exactly the kind [.ai/native_walk_design.md]'s tolerance policy
   is about, not something a hand-picked test value hides by luck). Direct
   primitives only, bypassing [Compute] entirely, so this is independent of
   whatever attention.ml happens to do. *)
let%expect_test
    "Direct: sdpa — split-sqrt scale is not bit-identical to a single multiply"
    =
  let scale = 3.7 and q = 12.25 and k = -5.5 in
  let sf = Direct.sqrt (Direct.const scale) in
  let split =
    Direct.mul (Direct.mul (Direct.const q) sf) (Direct.mul (Direct.const k) sf)
  in
  let unsplit =
    Direct.mul
      (Direct.mul (Direct.const q) (Direct.const k))
      (Direct.const scale)
  in
  Format.printf "split=%h unsplit=%h bit_equal=%b@." split unsplit
    (Core.Float_bits.equal_exact split unsplit);
  [%expect
    {| split=-0x1.f293333333333p+7 unsplit=-0x1.f293333333334p+7 bit_equal=false |}]

let%expect_test "Direct: sdpa — additive mask excludes the largest raw score" =
  (* q=1, k0=1 (dot=1), k1=5 (dot=5, the larger raw score); mask=[0,-inf]
     rules k1 out entirely, so the output is exactly value0. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 5.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 100. else 200.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c ->
        if chan c = 0 then 0. else Float.neg_infinity)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {100} |}]

let%expect_test "Direct: sdpa — additive negative mask (finite)" =
  (* q=0, so the raw dot product is 0 for every key regardless of k -- the
     mask alone decides the weights: mask=[-1,-3], diff 2, same shape as the
     default-scale case above but distinguished by its own values. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 7. else 9.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 9. else 11.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then -1. else -3.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {9.23841} |}]

let%expect_test "Direct: sdpa — batch and head extents independently" =
  (* Wk=1: the softmax is trivially one-hot on the sole key, so the output IS
     the value at every (D,H), which proves those two axes thread through
     independently rather than being conflated with each other or with W/C. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:2 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:2 ~w:1 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c ->
        float_of_int ((100 * bat c) + (10 * row c) + 1))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [D=2 H=2 W=1 C=1] {1, 11, 101, 111} |}]

let%expect_test "Direct: sdpa — large logits (max-subtraction stability)" =
  (* q=1, k0=1000, k1=999: raw scores 1000, 999. exp(1000) is not
     representable even in f64 (it is +inf); max-subtraction is what keeps
     this finite and correct: exp(0) + exp(-1). *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1000. else 999.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 10. else 20.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=1] {12.6894} |}]

let%expect_test
    "Direct: sdpa — an all -inf row yields 0, not NaN (_safe_softmax)" =
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key = Tensor.materialize key_shape (fun _ -> 1.) in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 7. else 9.)
  in
  let mask = Tensor.materialize mask_shape (fun _ -> Float.neg_infinity) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {0} |}]

let%expect_test "Direct: sdpa — Wq <> Wk" =
  (* Two query positions, three keys, all-zero dot products: both query
     positions get the same uniform 1/3 blend of the three values. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let query =
    Tensor.materialize query_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c -> [| 3.; 6.; 9. |].(col c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [W=2 C=1] {6, 6} |}]

let%expect_test "Direct: sdpa — mask broadcast on Wq (2D-equivalent shape)" =
  (* Two query positions share one mask row: mask stored with W=1 (broadcast)
     against a real Wq=2. Dot products are 0 everywhere, so the mask alone
     decides, identically for both query positions. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 5. else 7.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then 0. else -2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [W=2 C=1] {5.23841, 5.23841} |}]

let%expect_test "Direct: sdpa — mask broadcast on D (4D form)" =
  (* Two batch elements share one mask row: mask stored with D=1 (broadcast)
     against a real D=2. Values differ per batch (x10) to prove D itself
     still threads through correctly while the mask stays shared. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c ->
        let base = if col c = 0 then 5. else 7. in
        if bat c = 0 then base else base *. 10.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then 0. else -2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [D=2 H=1 W=1 C=1] {5.23841, 52.3841} |}]

(* op8-impl-review.md P1 (verified against source, fixed): [output_shape]
   checked D/H/C/W agreement across query/key/value but never N or T, and
   [Compute] reads key/value at the output coordinate's UNCHANGED N/T (never
   reduced through [broadcast_coord], exactly like D/H). A standalone or
   JSON-decoded graph with query.N=2, key.N=value.N=1 passed shape inference
   and then read key/value out of bounds evaluating output batch N=1; the
   reverse (key.N=2, query.N=1) silently ignored half of key/value. Both
   directions are now rejected here, the same as a D or H mismatch. *)
let%expect_test "Direct: sdpa — N and T must agree across query/key/value" =
  let ok = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let bad_n = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let bad_t = Vec6.shape ~n:1 ~t:2 ~d:1 ~h:1 ~w:1 ~c:1 in
  Format.printf "query.N=2, key.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:bad_n ~key_shape:ok
       ~value_shape:ok ~mask_shape:None);
  Format.printf "key.N=2, query.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:ok ~key_shape:bad_n
       ~value_shape:ok ~mask_shape:None);
  Format.printf "value.N=2, query.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:ok ~key_shape:ok
       ~value_shape:bad_n ~mask_shape:None);
  Format.printf "query.T=2, key.T=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:bad_t ~key_shape:ok
       ~value_shape:ok ~mask_shape:None);
  [%expect
    {|
    query.N=2, key.N=1: sdpa: N extent must agree (query vs key): 2 vs 1
    key.N=2, query.N=1: sdpa: N extent must agree (query vs key): 1 vs 2
    value.N=2, query.N=1: sdpa: N extent must agree (query vs value): 1 vs 2
    query.T=2, key.T=1: sdpa: T extent must agree (query vs key): 2 vs 1
    |}]

(* The total-work bound now includes N/T too (op8-impl-review.md P1's second
   half): without them, a graph could inflate real work by N*T while the
   bound only ever saw D*H*Wq*Wk*E*E. N=1024 alone reaches the same
   6-factor product the F12 counterexample used ([D=H=1, Wq=Wk=E=Ev=1024]),
   so folding N in must reject it too. *)
let%expect_test "Direct: sdpa — total-work bound counts N and T" =
  let big = Vec6.shape ~n:1024 ~t:1 ~d:1 ~h:1 ~w:1024 ~c:1024 in
  let big_kv = Vec6.shape ~n:1024 ~t:1 ~d:1 ~h:1 ~w:1024 ~c:1024 in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:big ~key_shape:big_kv
       ~value_shape:big_kv ~mask_shape:None);
  [%expect
    {| sdpa: total work N*T*D*H*Wq*Wk*E*E (score, row max and denominator are recomputed per output feature) exceeds 2147483648 after folding in the head-dimension extent E (running product 1073741824, this factor 1024) |}]
