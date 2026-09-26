(* [Core.Tagged_int]: the operations, and that applications stay independent at
   runtime as well as in the types (see tagged_int_opacity.t for the latter). *)

module A =
  Core.Tagged_int.Make
    (struct
      let prefix = "a"
    end)
    ()

module B =
  Core.Tagged_int.Make
    (struct
      let prefix = "b"
    end)
    ()

let%expect_test "equal, compare, succ, to_int" =
  let x = A.of_int 3 in
  Printf.printf "%b %b %d %d\n"
    (A.equal x (A.of_int 3))
    (A.equal x (A.succ x))
    (A.compare x (A.succ x))
    (A.to_int (A.succ x));
  [%expect {| true false -1 4 |}]

let%expect_test "the coercion is free" =
  let x = A.of_int 7 in
  Printf.printf "%d\n" ((x :> int) + 1);
  [%expect {| 8 |}]

let%expect_test "pp uses the prefix of its own application" =
  Format.printf "%a %a@." A.pp (A.of_int 12) B.pp (B.of_int 12);
  [%expect {| a12 b12 |}]

let%expect_test "Map and Set order by number" =
  let m =
    List.fold_left
      (fun m i -> A.Map.add (A.of_int i) (string_of_int i) m)
      A.Map.empty [ 5; 1; 3 ]
  in
  A.Map.iter (fun k v -> Format.printf "%a=%s " A.pp k v) m;
  let s = A.Set.of_list [ A.of_int 2; A.of_int 2; A.of_int 1 ] in
  Format.printf "| %d@." (A.Set.cardinal s);
  [%expect {| a1=1 a3=3 a5=5 | 2 |}]

let%expect_test "Next: alloc, alloc_n, after, reaches" =
  let a0, n = A.Next.alloc A.Next.first in
  let a1, n = A.Next.alloc n in
  let ids, n = A.Next.alloc_n n 3 in
  Format.printf "%a %a [%a] next=%a@." A.pp a0 A.pp a1
    (fun ppf ids -> List.iter (Format.fprintf ppf "%a;" A.pp) ids)
    ids A.Next.pp n;
  [%expect {| a0 a1 [a2;a3;a4;] next=a5 |}];
  (* [after] only ever raises the counter *)
  let n = A.Next.after (A.of_int 9) n in
  let m = A.Next.after (A.of_int 1) n in
  Format.printf "%a %a %b %b@." A.Next.pp n A.Next.pp m
    (A.Next.reaches n (A.of_int 9))
    (A.Next.reaches n (A.of_int 10));
  [%expect {| a10 a10 false true |}]

(* The id space is bounded before the addition, not after: a wrapped sum sails
   past the naive [next + count > max_int], and js_of_ocaml's [int] is 32 bits. *)
let%expect_test "Next.alloc_n: bounded before the addition" =
  let room next count =
    match A.Next.check_room (A.Next.of_int next) ~count with
    | () -> "ok"
    | exception Invalid_argument _ -> "exhausted"
  in
  Printf.printf "%s %s %s %s\n"
    (room (max_int - 3) 3)
    (room (max_int - 3) 4)
    (room 0 (-1)) (room max_int 0);
  [%expect {| ok exhausted exhausted ok |}]
