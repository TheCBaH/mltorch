(* Exercises Core.Monad.State: return, bind, map, get, and composed sequences.
   State is a plain int for all tests. *)

open Core.Monad.State

let run m s = m s

let%expect_test "return leaves state unchanged" =
  let v, s' = run (return 42) 0 in
  Printf.printf "value=%d state=%d\n" v s';
  [%expect {| value=42 state=0 |}]

let%expect_test "bind sequences computations and threads state" =
  let m =
    let* x = return 10 in
    let* y = return 20 in
    return (x + y)
  in
  let v, s' = run m 99 in
  Printf.printf "value=%d state=%d\n" v s';
  [%expect {| value=30 state=99 |}]

let%expect_test "get reads the current state" =
  let v, s' = run get 7 in
  Printf.printf "value=%d state=%d\n" v s';
  [%expect {| value=7 state=7 |}]

let%expect_test "map transforms value without touching state" =
  let m =
    let+ x = return 5 in
    x * x
  in
  let v, s' = run m 3 in
  Printf.printf "value=%d state=%d\n" v s';
  [%expect {| value=25 state=3 |}]

(* A simple counter monad built on Core.Monad.State to exercise real threading.
   [tick] increments the int state and returns the old count. *)
let tick s = (s, s + 1)

let%expect_test "tick counter: state advances, values carry old count" =
  let m =
    let* a = tick in
    let* b = tick in
    let* c = tick in
    return (a, b, c)
  in
  let (a, b, c), count = run m 0 in
  Printf.printf "ticks=(%d,%d,%d) final=%d\n" a b c count;
  [%expect {| ticks=(0,1,2) final=3 |}]

let%expect_test "get inside bind reads live state" =
  let m =
    let* s0 = get in
    let* s1 = get in
    return (s0, s1)
  in
  let (s0, s1), s' = run m 5 in
  Printf.printf "s0=%d s1=%d state=%d\n" s0 s1 s';
  [%expect {| s0=5 s1=5 state=5 |}]
