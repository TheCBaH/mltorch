let%expect_test "symint: build, pp, bind + eval" =
  let s0 = Symint.var "s0" in
  let e = Symint.(add (scale 2 s0) (const 1)) in
  Format.printf "%a@." Symint.pp e;
  [%expect {| 2*s0 + 1 |}];
  let env = Symint.(declare empty "s0" ~lo:1 ~hi:1024 ~divisor:8 ()) in
  let env = Symint.bind env "s0" 16 |> Core.or_raise Symint.pp_error in
  Format.printf "eval=%d@." (Option.get (Symint.eval env e));
  [%expect {| eval=33 |}]

let%expect_test "symint: bind rejects guard violations" =
  let env = Symint.(declare empty "s0" ~lo:1 ~hi:1024 ~divisor:8 ()) in
  let rejects v =
    match Symint.bind env "s0" v with Error _ -> true | Ok _ -> false
  in
  (* 17 not divisible by 8; 0 below lo; 24 valid *)
  Format.printf "%b %b %b@." (rejects 17) (rejects 0) (rejects 24);
  [%expect {| true true false |}]

let%expect_test "symint: size resolve (static vs sym), unbound stays None" =
  let env =
    Symint.(bind (declare empty "n" ~lo:1 ~hi:64 ()) "n" 8)
    |> Core.or_raise Symint.pp_error
  in
  let stat = Symint.Static 3 and dyn = Symint.Sym (Symint.var "n") in
  Format.printf "%a=%d %a=%d unbound=%b@." Symint.pp_size stat
    (Option.get (Symint.resolve env stat))
    Symint.pp_size dyn
    (Option.get (Symint.resolve env dyn))
    (Symint.resolve env (Symint.Sym (Symint.var "m")) = None);
  [%expect {| 3=3 n=8 unbound=true |}]
