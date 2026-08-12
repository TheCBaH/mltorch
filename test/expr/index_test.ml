(* Typed index expressions: the form-by-form port table, the division
   boundaries, and the checked-arithmetic bounds.

   These run under BOTH backends (see test/expr/dune), and js_of_ocaml's int is
   32 bits while the native one is 63. So nothing here prints a value that
   depends on the width -- the extremum cases are stated as oracles, and only
   their verdict is pinned. See .ai/js_backends_design.md. *)

open Expr

let origin = Coord.of_fn (fun _ -> 0)
let no_reducers _ = None

let pp_res fmt r =
  Core.Pretty.err_result ~ok:Fmt.int ~error:Eval.pp_index_error fmt r

let ev ?(output = origin) ?(reducers = no_reducers) e =
  Eval.index ~output ~reducers e

(* [Index.floor_div_pos] is result-returning, and its row is narrower than the
   evaluator's; widen rather than rebuild, so the detection backtrace survives. *)
let div build n d =
  match build (Index.const n) d with
  | Ok e -> ev e
  | Error e -> Error (e :> Eval.index_error Err.Error.t)

let%expect_test "every current index form has a typed equivalent" =
  (* The nine constructors of the representation this replaces, in order, each
     evaluated at a coordinate where the answer is not accidentally the same as
     its neighbours'. *)
  let output = Coord.of_fn (fun a -> Axis.to_int a * 10) in
  let show name e = Fmt.pr "%-16s %a@." name pp_res (ev ~output e) in
  show "output H" (Index.output Axis.H);
  show "zero" Index.zero;
  show "const" (Index.const 7);
  show "of_position" (Index.of_position (Index.output Axis.W));
  show "add" (Index.add (Index.const 7) (Index.const 5));
  show "scale" (Index.scale 3 (Index.const 5));
  show "min" (Index.min (Index.const 7) (Index.const 5));
  show "max" (Index.max (Index.const 7) (Index.const 5));
  show "clamp_low" (Index.clamp_low (Index.const (-4)));
  show "assume_position" (Index.assume_position (Index.const 4));
  [%expect
    {|
    output H         30
    zero             0
    const            7
    of_position      40
    add              12
    scale            15
    min              5
    max              7
    clamp_low        0
    assume_position  4
    |}];
  (* Division is the tenth and eleventh; it is result-returning, so it goes
     through [div]. *)
  Fmt.pr "floor %a   ceil %a@." pp_res
    (div Index.floor_div_pos 7 2)
    pp_res
    (div Index.ceil_div_pos 7 2);
  [%expect {| floor 3   ceil 4 |}]

let%expect_test "reducer lookup, bound and unbound" =
  let v, _ = Builder.run_from Builder.initial Builder.fresh_reduce in
  let bound x = if Reduce_var.equal x v then Some 5 else None in
  Fmt.pr "%a@." pp_res (ev ~reducers:bound (Index.reduce v));
  [%expect {| 5 |}];
  (* An unbound reducer is a structured failure, not a wrong answer: a free
     variable means the expression escaped its binder. *)
  Fmt.pr "%a@." pp_res (ev (Index.reduce v));
  [%expect {| unbound reducer #0 |}]

let%expect_test "floor/ceil division: floor and ceiling, never truncation" =
  (* Host integer division truncates toward zero, so the negative half is where
     a wrong implementation shows up. Values either side of +-multiples of 3. *)
  let row build =
    List.map
      (fun n -> Core.Pretty.to_string pp_res (div build n 3))
      [ -7; -6; -5; -4; -3; -2; -1; 0; 1; 2; 3; 4; 5; 6; 7 ]
    |> String.concat " "
  in
  Fmt.pr "n     -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7@.";
  Fmt.pr "floor %s@." (row Index.floor_div_pos);
  Fmt.pr "ceil  %s@." (row Index.ceil_div_pos);
  [%expect
    {|
    n     -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7
    floor -3 -2 -2 -2 -1 -1 -1 0 0 0 1 1 1 2 2
    ceil  -2 -2 -1 -1 -1 0 0 0 1 1 1 2 2 2 3
    |}];
  (* Divide by one folds to the operand -- the one identity kept from the start,
     because the representation this replaces already performs it. *)
  Fmt.pr "%a %a@." pp_res
    (div Index.floor_div_pos (-7) 1)
    pp_res
    (div Index.ceil_div_pos (-7) 1);
  [%expect {| -7 -7 |}]

let%expect_test "a non-positive divisor never reaches the AST" =
  Fmt.pr "%a@." pp_res (div Index.floor_div_pos 10 0);
  [%expect {| divisor must be > 0, got 0 |}];
  Fmt.pr "%a@." pp_res (div Index.ceil_div_pos 10 (-2));
  [%expect {| divisor must be > 0, got -2 |}]

let%expect_test
    "division at the representation boundary is defined, not an overflow" =
  (* For a positive divisor every floor and ceiling quotient of every
     representable dividend is itself representable, so the remainder-adjusted
     form has no error case here. The form it replaces negates the dividend and
     would manufacture one. Backend-independent: stated as an oracle. *)
  let ok n d build =
    match div build n d with Ok _ -> true | Error _ -> false
  in
  let extremes =
    [
      (Stdlib.min_int, 1);
      (Stdlib.min_int, 2);
      (Stdlib.max_int, 1);
      (Stdlib.max_int, 3);
    ]
  in
  Fmt.pr "all defined: %b@."
    (List.for_all
       (fun (n, d) -> ok n d Index.floor_div_pos && ok n d Index.ceil_div_pos)
       extremes);
  [%expect {| all defined: true |}];
  (* And by 1 the answer is the dividend itself, at both extrema. *)
  let same n build =
    match div build n 1 with Ok v -> v = n | Error _ -> false
  in
  Fmt.pr "identity at extrema: %b@."
    (List.for_all
       (fun n -> same n Index.floor_div_pos && same n Index.ceil_div_pos)
       [ Stdlib.min_int; Stdlib.max_int ]);
  [%expect {| identity at extrema: true |}]

let%expect_test "arithmetic is bounds-checked before the operation, not after" =
  let errs e = match ev e with Ok _ -> false | Error _ -> true in
  let big = Index.const Stdlib.max_int and small = Index.const Stdlib.min_int in
  (* max_int + 1 and min_int - 1: the classic wrap either way. *)
  Fmt.pr "add overflows: %b %b@."
    (errs (Index.add big (Index.const 1)))
    (errs (Index.add small (Index.const (-1))));
  [%expect {| add overflows: true true |}];
  (* min_int * -1. This is the case a post-hoc [r / a = b] check misses: the
     product wraps to min_int and min_int / -1 is min_int too, so the equality
     holds on a genuine overflow. *)
  Fmt.pr "min_int * -1 overflows: %b@." (errs (Index.scale (-1) small));
  [%expect {| min_int * -1 overflows: true |}];
  Fmt.pr "max_int * 2 overflows: %b@." (errs (Index.scale 2 big));
  [%expect {| max_int * 2 overflows: true |}];
  (* Multiplication that fits must still succeed, including at the edges -- a
     check that rejects everything is not a bound either. *)
  let fits e = match ev e with Ok _ -> true | Error _ -> false in
  Fmt.pr "in range still ok: %b %b %b@."
    (fits (Index.scale 1 small))
    (fits (Index.scale (-1) big))
    (fits (Index.scale 0 small));
  [%expect {| in range still ok: true true true |}]

let%expect_test "an intermediate cannot wrap back into range" =
  (* The reason the check is per-operation rather than on the final value:
     (max_int + max_int) + min_int is representable as a mathematical integer
     result, but the first addition already left the domain. Checking only the
     total would accept it. *)
  let e =
    Index.add
      (Index.add (Index.const Stdlib.max_int) (Index.const Stdlib.max_int))
      (Index.const Stdlib.min_int)
  in
  Fmt.pr "rejected at the inner add: %b@."
    (match ev e with Ok _ -> false | Error _ -> true);
  [%expect {| rejected at the inner add: true |}]

let%expect_test "float_of_index: exactness is a round trip, not a threshold" =
  let exact i =
    match Eval.float_of_index i with Ok _ -> true | Error _ -> false
  in
  let oracle i =
    Int64.equal (Int64.of_float (Stdlib.float_of_int i)) (Int64.of_int i)
  in
  let sample =
    if Sys.int_size > 53 then
      let p53 = 1 lsl 53 and p54 = 1 lsl 54 in
      [
        0;
        1;
        -1;
        p53 - 1;
        p53;
        p53 + 1;
        p53 + 2;
        p54;
        p54 + 2;
        -p53 - 1;
        -p54 - 2;
        Stdlib.min_int;
        Stdlib.max_int;
      ]
    else [ 0; 1; -1; 1000; Stdlib.min_int; Stdlib.max_int ]
  in
  Fmt.pr "agrees with the oracle everywhere: %b@."
    (List.for_all (fun i -> exact i = oracle i) sample);
  [%expect {| agrees with the oracle everywhere: true |}];
  (* The distinguishing cases. A plain "reject above 2^53" rule gets these wrong
     in BOTH directions: 2^53+2, 2^54 and min_int are exact while 2^53+1,
     2^54+2 and max_int are not. Under js_of_ocaml there is nothing above 2^53
     to test and the meaningful claim is that everything representable is
     exact -- which is also why float_of_index short-circuits there rather than
     running two emulated Int64 conversions per pixel. *)
  let verdicts =
    if Sys.int_size > 53 then
      let p53 = 1 lsl 53 and p54 = 1 lsl 54 in
      [
        exact p53;
        not (exact (p53 + 1));
        exact (p53 + 2);
        exact p54;
        not (exact (p54 + 2));
        exact Stdlib.min_int;
        not (exact Stdlib.max_int);
      ]
    else [ exact Stdlib.min_int; exact Stdlib.max_int; exact 0; exact (-1) ]
  in
  Fmt.pr "distinguishing cases: %b@." (List.for_all Fun.id verdicts);
  [%expect {| distinguishing cases: true |}]

let%expect_test "printed form matches the representation it replaces" =
  let names v = Fmt.str "r%d" (Reduce_var.hash v) in
  let show e = Fmt.pr "%a@." (Pp.index ~names) e in
  show (Index.output Axis.H);
  show Index.zero;
  show
    (Index.add
       (Index.scale 2 (Index.of_position (Index.output Axis.W)))
       (Index.const 3));
  show (Index.min (Index.const 2) (Index.max (Index.const 0) (Index.const 5)));
  (* [clamp_low] is its own node now but must still render as max(0,_), and
     [of_position]/[assume_position] stay transparent -- all three were spelled
     that way before, and a change here would move every conv and pool golden. *)
  show (Index.clamp_low (Index.const (-1)));
  show (Index.assume_position (Index.const 4));
  [%expect
    {|
    H
    0
    2*W+3
    min(2,max(0,5))
    max(0,-1)
    4
    |}];
  Fmt.pr "%a@."
    (Core.Pretty.result ~ok:(Pp.index ~names)
       ~error:(Core.Pretty.error_kind Index.pp_error))
    (Index.floor_div_pos (Index.of_position (Index.output Axis.H)) 2);
  [%expect {| floor_div(H,2) |}]

let%expect_test "smart constructors fold only exact integer identities" =
  let names v = Fmt.str "r%d" (Reduce_var.hash v) in
  let show e = Fmt.pr "%a@." (Pp.index ~names) e in
  let h = Index.of_position (Index.output Axis.H) in
  (* Deferred until after the migration on purpose: landing these earlier would
     have mixed a readability change into the cutover's diff, whose whole point
     was that nothing observable moved. *)
  show (Index.scale 1 h);
  show (Index.add h (Index.const 0));
  show (Index.add (Index.const 0) h);
  show (Index.scale 7 (Index.const 0));
  show (Index.add (Index.scale 2 (Index.const 0)) h);
  [%expect {|
    H
    H
    H
    0
    H
    |}];
  (* And nothing else: a non-unit scale, a non-zero addend and a division all
     survive, or the fold would be changing the expression rather than tidying
     its spelling. *)
  show (Index.scale 2 h);
  show (Index.add h (Index.const 3));
  show (Index.scale (-1) h);
  [%expect {|
    2*H
    H+3
    -1*H
    |}];
  (* Folding is exact at the extremes too -- [scale k 0] is 0 for every [k],
     including ones whose product would otherwise overflow. *)
  Fmt.pr "%a %a@." (Pp.index ~names)
    (Index.scale Stdlib.min_int (Index.const 0))
    pp_res
    (ev (Index.scale Stdlib.min_int (Index.const 0)));
  [%expect {| 0 0 |}]
