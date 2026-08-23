(* Values, reductions, intrinsics: construction, printing, structural identity
   and validation. Runs under both backends (see test/expr/dune). *)

open Expr

let src n = Source.create n
let out = Coord.of_fn (fun a -> Index.output a)
let pp_v = Pp.value

(* A conv-shaped body: a nested pair of reductions over a load, which is the
   shape every scope question below is really about. *)
let nested ~kind =
  let open Builder.Syntax in
  Builder.reduction ~kind ~lo:Index.zero ~hi:(Index.const 2) (fun r1 ->
      Builder.reduction ~kind ~lo:Index.zero ~hi:(Index.const 3) (fun r2 ->
          let+ () = Builder.return () in
          Value.mul
            (Value.load (src 0) (Coord.set (Coord.set out Axis.H r1) Axis.W r2))
            (Value.value_of_index (Index.of_position r2))))

let%expect_test "printed form matches the representation being replaced" =
  let x = Value.load (src 0) out in
  let y = Value.load (src 1) out in
  (* relu, as the ops derive it: select + lt, no dedicated primitive. *)
  Fmt.pr "%a@." pp_v
    (Value.select (Bool.value_lt x (Value.const 0.)) (Value.const 0.) x);
  [%expect {| select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C]) |}];
  Fmt.pr "%a@." pp_v (Value.add x y);
  [%expect {| (t0[N,T,D,H,W,C] + t1[N,T,D,H,W,C]) |}];
  Fmt.pr "%a@." pp_v (Value.sqrt (Value.div x (Value.const 2.)));
  [%expect {| sqrt((t0[N,T,D,H,W,C] / 2)) |}];
  Fmt.pr "%a@." pp_v (Value.erf (Value.div x (Value.const 2.)));
  [%expect {| erf((t0[N,T,D,H,W,C] / 2)) |}];
  Fmt.pr "%a@." pp_v (Builder.run (nested ~kind:Reduction.Sum));
  [%expect
    {| sum(r1=0..2: sum(r2=0..3: (t0[N,T,D,r1,r2,C] * value_of_index(r2)))) |}];
  (* [Round_f32] is the one genuinely new node: it has no spelling in the old
     representation to preserve. *)
  Fmt.pr "%a@." pp_v (Value.round_f32 (Value.add x y));
  [%expect {| f32((t0[N,T,D,H,W,C] + t1[N,T,D,H,W,C])) |}]

let%expect_test "reducer names are lexical, not allocation order" =
  (* Two structurally identical formulas from supplies at different points must
     print identically -- otherwise every golden would depend on how many
     reductions happened to be built earlier in the program. *)
  let a = Builder.run (nested ~kind:Reduction.Sum) in
  let advanced, _ =
    Builder.run_from Builder.initial
      (Builder.bind Builder.fresh_reduce (fun _ ->
           Builder.bind Builder.fresh_reduce (fun _ ->
               nested ~kind:Reduction.Sum)))
  in
  Fmt.pr "same printed form: %b@."
    (String.equal
       (Core.Pretty.to_string pp_v a)
       (Core.Pretty.to_string pp_v advanced));
  [%expect {| same printed form: true |}];
  (* And they are alpha-equivalent, so they compare and hash equally too. *)
  Fmt.pr "equal: %b   same hash: %b@." (Value.equal a advanced)
    (Value.hash a = Value.hash advanced);
  [%expect {| equal: true   same hash: true |}];
  (* Lexical means SCOPED, not one map keyed by identity. Two fragments built
     from independent supplies both bind ordinal 0; in sibling scopes that is
     well scoped -- neither shadows the other, and [Check] accepts it -- but a
     global identity-to-name map keeps only the last occurrence and prints both
     binders as r2. *)
  let frag =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun r -> Builder.return (Value.value_of_index (Index.of_position r))))
  in
  let siblings = Value.add frag frag in
  Fmt.pr "%a@." pp_v siblings;
  [%expect
    {|
    (sum(r1=0..2: value_of_index(r1)) + sum(r2=0..2: value_of_index(r2)))
    |}];
  Fmt.pr "well scoped: %a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error)
    (Check.value siblings);
  [%expect {| well scoped: ok |}];
  (* And a free reference must not pick up a sibling's binder: it prints as the
     opaque identity, even though the twin on the left binds exactly that one. *)
  let zeroth, _ = Builder.run_from Builder.initial Builder.fresh_reduce in
  Fmt.pr "%a@." pp_v
    (Value.add frag
       (Value.value_of_index (Index.of_position (Index.reduce zeroth))));
  [%expect {| (sum(r1=0..2: value_of_index(r1)) + value_of_index(?#0)) |}]

let%expect_test "structural identity separates what it must" =
  let base = Builder.run (nested ~kind:Reduction.Sum) in
  let differs f = not (Value.equal base (Builder.run (f ()))) in
  (* A different reduction kind, a bound off by one, and a different source are
     all genuinely different expressions -- alpha-equivalence must not swallow
     them. *)
  Fmt.pr "kind: %b@." (differs (fun () -> nested ~kind:Reduction.Max));
  [%expect {| kind: true |}];
  let off_by_one =
    Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
      (fun r1 ->
        Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 4)
          (fun r2 ->
            Builder.return
              (Value.mul
                 (Value.load (src 0)
                    (Coord.set (Coord.set out Axis.H r1) Axis.W r2))
                 (Value.value_of_index (Index.of_position r2)))))
  in
  Fmt.pr "upper bound: %b@." (not (Value.equal base (Builder.run off_by_one)));
  [%expect {| upper bound: true |}];
  (* Operand order matters: the language's arithmetic is not reassociable. *)
  let x = Value.load (src 0) out and y = Value.load (src 1) out in
  Fmt.pr "operand order: %b@."
    (not (Value.equal (Value.sub x y) (Value.sub y x)));
  [%expect {| operand order: true |}]

let%expect_test "constants distinguish signed zero and canonicalize NaNs" =
  (* Ordinary comparison equates -0. with 0. Structural identity must retain
     that distinction. NaN payloads, however, cannot survive a portable round
     trip through a JavaScript Number, so structural identity canonicalizes
     them. *)
  let c = Value.const in
  Fmt.pr "0. vs -0.: %b@." (Value.equal (c 0.) (c (-0.)));
  [%expect {| 0. vs -0.: false |}];
  let nan1 = Int64.float_of_bits 0x7ff8000000000001L in
  let nan2 = Int64.float_of_bits 0x7ff8000000000002L in
  Fmt.pr "two NaN payloads: %b@." (Value.equal (c nan1) (c nan2));
  [%expect {| two NaN payloads: true |}];
  Fmt.pr "a NaN with itself: %b@." (Value.equal (c nan1) (c nan1));
  [%expect {| a NaN with itself: true |}];
  (* Hashing follows the same canonical representation. *)
  Fmt.pr "two NaNs have the same hash: %b@."
    (Value.hash (c nan1) = Value.hash (c nan2));
  [%expect {| two NaNs have the same hash: true |}];
  Fmt.pr "hashes differ for +-0.: %b@."
    (Value.hash (c 0.) <> Value.hash (c (-0.)));
  [%expect {| hashes differ for +-0.: true |}]

let%expect_test "Fold: scope-aware queries" =
  let e = Builder.run (nested ~kind:Reduction.Sum) in
  Fmt.pr "size %d  depth %d  intrinsics %d  assume_sites %d@." (Fold.size e)
    (Fold.depth e) (Fold.intrinsics e) (Fold.assume_sites e);
  [%expect {| size 17  depth 6  intrinsics 0  assume_sites 0 |}];
  Fmt.pr "sources %a@."
    (Fmt.list ~sep:(Fmt.any ",") Source.pp)
    (Source.Set.elements (Fold.sources e));
  [%expect {| sources t0 |}];
  (* Only N/T/D/C survive: H and W were replaced by the reducers. *)
  Fmt.pr "output axes %a@."
    (Fmt.list ~sep:(Fmt.any ",") Axis.pp)
    (Fold.output_axes e);
  [%expect {| output axes N,T,D,C |}];
  (* A well-formed expression has no free reducer; the body alone, lifted out
     of its binders, has two. *)
  Fmt.pr "free at top: %d@." (Reduce_var.Set.cardinal (Fold.free_reducers e));
  [%expect {| free at top: 0 |}];
  (* Index trees count toward both. They are where a load's addressing lives,
     so a limit that treated them as leaves would bound nothing useful: this
     expression is a single value node over a twenty-deep affine index, and
     reporting size=1 depth=1 for it would make Check's limits meaningless. *)
  let deep =
    Value.value_of_index
      (List.fold_left
         (fun acc n -> Index.add acc (Index.const n))
         (Index.of_position (Index.output Axis.H))
         (List.init 20 (fun i -> i + 1)))
  in
  Fmt.pr "deep index: size %d depth %d@." (Fold.size deep) (Fold.depth deep);
  [%expect {| deep index: size 43 depth 23 |}];
  Fmt.pr "assume_position is locatable: %d@."
    (Fold.assume_sites
       (Value.value_of_index
          (Index.of_position (Index.assume_position (Index.const 3)))));
  [%expect {| assume_position is locatable: 1 |}]

let%expect_test "Check rejects what composition can still break" =
  let ok = Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error in
  let e = Builder.run (nested ~kind:Reduction.Sum) in
  Fmt.pr "%a@." ok (Check.value e);
  [%expect {| ok |}];
  (* A free reducer: a body that escaped its binder. Built by taking a variable
     from one supply and never binding it. *)
  let stray, _ = Builder.run_from Builder.initial Builder.fresh_reduce in
  Fmt.pr "%a@." ok
    (Check.value
       (Value.value_of_index (Index.of_position (Index.reduce stray))));
  [%expect {| free reducer #0 |}];
  (* A duplicate binder: TWO independently built fragments composed without
     freshening. Both supplies start at [initial], so both mint ordinal 0 and
     the inner binder captures references meant for the outer one. This is the
     real-world shape of capture, not a contrived one. *)
  let frag () =
    Builder.run_from Builder.initial
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun r -> Builder.return (Value.value_of_index (Index.of_position r))))
  in
  let inner, _ = frag () in
  let captured, _ =
    Builder.run_from Builder.initial
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun _ -> Builder.return inner))
  in
  Fmt.pr "%a@." ok (Check.value captured);
  [%expect {| reducer #0 is bound twice on one path |}];
  (* Size and depth limits are opt-in, and report the LIMIT rather than the
     measure -- naming the actual size would mean measuring the whole tree,
     which is what the limit exists to avoid. Boundaries are exact: [Fold.size e]
     is 17 and [Fold.depth e] is 6, so 17 and 6 pass and 16 and 5 do not. *)
  Fmt.pr "%a  %a@." ok
    (Check.value ~max_size:3 e)
    ok
    (Check.value ~max_depth:2 e);
  [%expect {| size exceeds limit 3  depth exceeds limit 2 |}];
  Fmt.pr "at the limit: %a %a   one under: %a %a@." ok
    (Check.value ~max_size:17 e)
    ok
    (Check.value ~max_depth:6 e)
    ok
    (Check.value ~max_size:16 e)
    ok
    (Check.value ~max_depth:5 e);
  [%expect
    {| at the limit: ok ok   one under: size exceeds limit 16 depth exceeds limit 5 |}]

let%expect_test "the limits are metered, so they survive the tree they reject" =
  (* Both are checked by METERING the traversal, not by measuring and then
     comparing. Measuring first means recursing the full tree, which exhausts
     the stack on exactly the oversized input the limit exists to refuse -- a
     guard that only works on trees that did not need one. Restore
     [Fold.depth e > limit] and this goes red with [Stack_overflow] on BOTH
     backends -- a million levels is chosen to clear the native stack, and JS's
     is far smaller. Building the tree is iterative, so nothing but the check
     under test recurses.

     The depth lives in the INDEX tree, which the value walk reaches through a
     single [Value_of_index]: one value node over a very deep address. *)
  let ok = Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error in
  let deep =
    let rec build n acc =
      if n = 0 then acc else build (n - 1) (Index.add acc (Index.const n))
    in
    Value.value_of_index (build 1_000_000 (Index.const 0))
  in
  Fmt.pr "%a  %a@." ok
    (Check.value ~max_depth:10 deep)
    ok
    (Check.value ~max_size:10 deep);
  [%expect {| depth exceeds limit 10  size exceeds limit 10 |}];
  (* And both budgets must ride ONE traversal. Checking them in sequence is not
     enough: whichever walk runs first descends the whole input whenever its own
     bound is loose, so a size limit too loose to reject anything still walks a
     million levels before the depth limit is consulted -- and swapping the
     order breaks the dual case instead. Each limit below is unreachable in one
     direction and tight in the other. *)
  Fmt.pr "%a  %a@." ok
    (Check.value ~max_size:Stdlib.max_int ~max_depth:10 deep)
    ok
    (Check.value ~max_size:10 ~max_depth:Stdlib.max_int deep);
  [%expect {| depth exceeds limit 10  size exceeds limit 10 |}];
  (* Metering must not turn WIDE into deep: a balanced tree of thousands of
     nodes is only a dozen levels deep, and a depth meter that charged its
     budget per node rather than per level would reject it. *)
  let rec balanced n =
    if n <= 1 then Value.const 1.
    else Value.add (balanced (n / 2)) (balanced (n - (n / 2)))
  in
  let wide = balanced 2_000 in
  Fmt.pr "size %d depth %d: %a@." (Fold.size wide) (Fold.depth wide) ok
    (Check.value ~max_depth:11 wide);
  [%expect {| size 3999 depth 12: depth exceeds limit 11 |}];
  Fmt.pr "%a@." ok (Check.value ~max_depth:12 wide);
  [%expect {| ok |}]

let%expect_test "max-pool intrinsic: descriptor, geometry, printing" =
  let mk ?(in_h = 4) ?(in_w = 4) ?(kernel = 2) ?(stride = 1) ?(pad = 1) result =
    Intrinsic.max_pool ~source:(src 0) ~in_h ~in_w ~kernel_h:kernel
      ~kernel_w:kernel ~stride_h:stride ~stride_w:stride ~pad_h:pad ~pad_w:pad
      ~out ~result
  in
  let pp_i =
    Core.Pretty.err_result
      ~ok:(fun fmt d -> pp_v fmt (Value.intrinsic d))
      ~error:Intrinsic.pp_error
  in
  Fmt.pr "%a@." pp_i (mk Intrinsic.Max_pool.Value);
  [%expect {| max_pool2d_value(t0; k=2x2 s=1x1 p=1x1; out=[N,T,D,H,W,C]) |}];
  Fmt.pr "%a@." pp_i (mk Intrinsic.Max_pool.Index);
  [%expect {| max_pool2d_index(t0; k=2x2 s=1x1 p=1x1; out=[N,T,D,H,W,C]) |}];
  (* Invalid geometry never becomes a descriptor. *)
  Fmt.pr "%a@." pp_i (mk ~stride:0 Intrinsic.Max_pool.Value);
  [%expect {| stride_h must be > 0, got 0 |}];
  Fmt.pr "%a@." pp_i (mk ~pad:(-1) Intrinsic.Max_pool.Value);
  [%expect {| pad_h must be >= 0, got -1 |}]

let%expect_test "max-pool geometry: the window both interpreters share" =
  let d =
    match
      Intrinsic.max_pool ~source:(src 0) ~in_h:4 ~in_w:4 ~kernel_h:2 ~kernel_w:2
        ~stride_h:1 ~stride_w:1 ~pad_h:1 ~pad_w:1 ~out
        ~result:Intrinsic.Max_pool.Value
    with
    | Ok d -> d
    | Error _ -> assert false
  in
  let show out_h out_w =
    match Intrinsic.window d ~out_h ~out_w with
    | Ok w ->
        Fmt.pr "(%d,%d) -> h[%d,%d) w[%d,%d)@." out_h out_w
          w.Intrinsic.Window.hlo w.Intrinsic.Window.hhi w.Intrinsic.Window.wlo
          w.Intrinsic.Window.whi
    | Error e ->
        Fmt.pr "(%d,%d) -> %a@." out_h out_w Intrinsic.pp_error
          (Err.Error.kind e)
  in
  (* Padding clips at the low edge, the extent clips at the high edge -- the
     same bounds the current implementation computes. *)
  show 0 0;
  show 1 1;
  show 4 4;
  [%expect
    {|
    (0,0) -> h[0,1) w[0,1)
    (1,1) -> h[0,2) w[0,2)
    (4,4) -> h[3,4) w[3,4)
    |}];
  Fmt.pr "flat (2,3) = %a@."
    (Core.Pretty.err_result ~ok:Fmt.int ~error:Intrinsic.pp_error)
    (Intrinsic.flat_index d ~ih:2 ~iw:3);
  [%expect {| flat (2,3) = 11 |}];
  (* The geometry is CHECKED, not just the descriptor fields: out_h * stride is
     an aggregate of individually valid factors and can still leave the domain.
     This is what native's Ground_eval gets for free by calling the same helper
     instead of recomputing the products itself. *)
  Fmt.pr "aggregate overflow caught: %b@."
    (match Intrinsic.window d ~out_h:Stdlib.max_int ~out_w:0 with
    | Ok _ -> false
    | Error _ -> true);
  [%expect {| aggregate overflow caught: true |}]

let%expect_test "Round_f32 reference: the f32 storage round trip" =
  (* The reference is [Int32.float_of_bits (Int32.bits_of_float x)]. The oracle
     it is checked against must be INDEPENDENT: Walk_core.Float32.to_f32 is the
     same expression, so comparing to it would prove nothing. An actual f32
     Bigarray round trip goes through the runtime's own narrowing instead. *)
  let round x = Int32.float_of_bits (Int32.bits_of_float x) in
  let stored x =
    let a = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
    Bigarray.Array1.set a 0 x;
    Bigarray.Array1.get a 0
  in
  let cases =
    [ 0.; -0.; 1.; 0.1; 1e-45; 3.4e38; Float.infinity; -1.5e-8; 16777217. ]
  in
  Fmt.pr "agrees with an f32 store/load: %b@."
    (List.for_all
       (fun x -> Core.Float_bits.equal_exact (round x) (stored x))
       cases);
  [%expect {| agrees with an f32 store/load: true |}];
  (* 0.1 is the case that matters: it is not f32-representable, so a no-op
     implementation would pass every other line here. *)
  Fmt.pr "0.1 actually changes: %b@." (round 0.1 <> 0.1);
  [%expect {| 0.1 actually changes: true |}];
  Fmt.pr "idempotent: %b@." (round (round 0.1) = round 0.1);
  [%expect {| idempotent: true |}]
